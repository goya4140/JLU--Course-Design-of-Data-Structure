/*
 * ============================================================
 *  五子棋 AI —— v6 SOTA 终稿
 *  平台：Botzone (Gomoku-Swap1)
 *  编译：g++ -O2 -std=c++17 -o gomoku Gomoku_v6_sota.cpp
 *
 *  I/O 格式（与 Gomoku_simple.cpp 完全一致的简单交互模式）：
 *    输入：
 *      n                      <- 当前回合数
 *      x_opp1 y_opp1          <- 第1回合对方落子
 *      x_my1  y_my1           <- 第1回合本方落子
 *      ...（共 n-1 对）
 *      x_oppN y_oppN          <- 第n回合对方最新落子
 *    输出：new_x new_y         <- 本方决策（-1 -1 仅在第1回合后手表示换手）
 *
 *  棋盘编码（与 v5 完全一致）：
 *    board[x][y] = 0   → 空
 *    board[x][y] = P1  → 本方棋子 (1)
 *    board[x][y] = P2  → 对方棋子 (2)
 *
 *  v6 新增四大核心技术（相对 v5）：
 *  ┌────┬──────────────────────────────────┬──────────────────────────┐
 *  │优先│ 技术                             │ 论文来源                 │
 *  ├────┼──────────────────────────────────┼──────────────────────────┤
 *  │ 1  │ VCF 连续冲四胜利搜索（带时限）    │ Allis et al., AI 1993    │
 *  │ 2  │ MCTS-Solver 必胜/必败标记传播     │ Winands et al., CG 2008  │
 *  │ 3  │ Zobrist 哈希 + 置换表            │ Zobrist, Tech Rep. 1970  │
 *  │ 4  │ 历史启发 + Killer Move 排序       │ Schaeffer, TPAMI 1989   │
 *  └────┴──────────────────────────────────┴──────────────────────────┘
 *
 *  v6 相对 v5 的主要 Bug 修复：
 *    ① VCF 加时间限制防超时
 *    ② MCTS 改用落子/撤销方案（不再使用 replayPath），避免棋盘状态污染
 *    ③ UCB1 公式修正：wins/visits（v5 误用 (visits-wins)/visits）
 *    ④ VCF 正确处理"双冲四"（v5 仅检测活四，漏掉了双冲四的必胜情况）
 *
 *  整体决策框架（5 层漏斗）：
 *    层1 即时胜利/防守
 *    层2 双威胁战术判断
 *    层3 VCF 必杀搜索（带时间限制）
 *    层4 迭代加深 Alpha-Beta（Zobrist TT + 历史启发）
 *    层5 MCTS（MCTS-Solver 增强，落子/撤销方案）
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ─── 常量 ────────────────────────────────────────────────────
static const int SIZE  = 15;
static const int EMPTY = 0;
static const int P1    = 1;   // 本方（AI）
static const int P2    = 2;   // 对方
static const int INF   = 1e8;

// ─── 计时（使用 clock()，与 Botzone 官方示例一致）────────────
static clock_t g_start;
inline double elapsed() {
    return (double)(clock() - g_start) / CLOCKS_PER_SEC;
}
// 总时间预算：0.90 秒（Botzone 限时 1 秒，留余量）
static const double TL = 0.90;

// ─── 棋盘 ────────────────────────────────────────────────────
static int board[SIZE][SIZE];

// ─── 4 个主方向向量 ──────────────────────────────────────────
static const int DX[4] = {0, 1, 1,  1};
static const int DY[4] = {1, 0, 1, -1};

// ═══════════════════════════════════════════════════════════════
//  §1  棋形评估
// ═══════════════════════════════════════════════════════════════
static const int SC_FIVE = 1000000;
static const int SC_L4   =   50000; // 活四（下一步必赢）
static const int SC_R4   =   10000; // 冲四（对手唯一强制应答）
static const int SC_L3   =    5000;
static const int SC_R3   =     500;
static const int SC_L2   =     200;
static const int SC_R2   =      20;

// 单方向棋形评分（假设 board[x][y] 已落 color）
static int scoreDir(int x, int y, int dx, int dy, int color) {
    int cnt = 1, ol = 0, or_ = 0;
    for (int k = 1; k <= 4; k++) {
        int nx=x+k*dx, ny=y+k*dy;
        if (nx<0||nx>=SIZE||ny<0||ny>=SIZE) break;
        if (board[nx][ny]==color) cnt++;
        else { if (board[nx][ny]==EMPTY) or_=1; break; }
    }
    for (int k = 1; k <= 4; k++) {
        int nx=x-k*dx, ny=y-k*dy;
        if (nx<0||nx>=SIZE||ny<0||ny>=SIZE) break;
        if (board[nx][ny]==color) cnt++;
        else { if (board[nx][ny]==EMPTY) ol=1; break; }
    }
    int open = ol + or_;
    if (cnt>=5) return SC_FIVE;
    if (cnt==4) return open==2 ? SC_L4 : (open==1 ? SC_R4 : 0);
    if (cnt==3) return open==2 ? SC_L3 : (open==1 ? SC_R3 : 0);
    if (cnt==2) return open==2 ? SC_L2 : (open==1 ? SC_R2 : 0);
    return 0;
}

// 4 个方向综合威胁分
static int analyzePoint(int x, int y, int color) {
    int s = 0;
    for (int d=0; d<4; d++) s += scoreDir(x, y, DX[d], DY[d], color);
    return s;
}

// 全局棋面静态估值（P1 视角）
static int evaluateBoard() {
    int s1=0, s2=0;
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++) {
            if (board[i][j]==P1) s1+=analyzePoint(i,j,P1);
            else if (board[i][j]==P2) s2+=analyzePoint(i,j,P2);
        }
    return s1-s2;
}

// 检测 (x,y) 落子 color 后是否连五
static bool checkWin(int x, int y, int color) {
    for (int d=0; d<4; d++) {
        int cnt=1;
        for (int s:{1,-1}) {
            int nx=x, ny=y;
            while (true) {
                nx+=s*DX[d]; ny+=s*DY[d];
                if (nx<0||nx>=SIZE||ny<0||ny>=SIZE||board[nx][ny]!=color) break;
                cnt++;
            }
        }
        if (cnt>=5) return true;
    }
    return false;
}

// 邻近剪枝：只考虑距已有棋子 radius 格内的空点
static bool hasNeighbor(int x, int y, int radius=2) {
    for (int dx=-radius;dx<=radius;dx++)
        for (int dy=-radius;dy<=radius;dy++) {
            if (!dx&&!dy) continue;
            int nx=x+dx, ny=y+dy;
            if (nx>=0&&nx<SIZE&&ny>=0&&ny<SIZE&&board[nx][ny]!=EMPTY) return true;
        }
    return false;
}

// ═══════════════════════════════════════════════════════════════
//  §2  走法生成
// ═══════════════════════════════════════════════════════════════
struct Move { int x, y, score; };

// 找 color 的立即赢棋点（落下即连五）
static bool getWinMove(int color, int &wx, int &wy) {
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++) {
            if (board[i][j]!=EMPTY) continue;
            board[i][j]=color;
            bool w=checkWin(i,j,color);
            board[i][j]=EMPTY;
            if (w) { wx=i; wy=j; return true; }
        }
    return false;
}

// 生成带评分的候选走法（降序，最多 topN 个）
static vector<Move> genMoves(int color, int histSc[][SIZE]=nullptr, int topN=20) {
    int opp=3-color;
    vector<Move> mv; mv.reserve(30);
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++) {
            if (board[i][j]!=EMPTY||!hasNeighbor(i,j)) continue;
            int s=analyzePoint(i,j,color)+analyzePoint(i,j,opp);
            if (histSc) s+=histSc[i][j]*8;
            mv.push_back({i,j,s});
        }
    sort(mv.begin(),mv.end(),[](const Move&a,const Move&b){return a.score>b.score;});
    if ((int)mv.size()>topN) mv.resize(topN);
    return mv;
}

// ═══════════════════════════════════════════════════════════════
//  §3  VCF 连续冲四胜利搜索（带时间限制）
//
//  【论文】Allis, L.V., et al. (1993). "Go-Moku and Threat-Space Search."
//          Artificial Intelligence, 56(1), 1-37.
//
//  【核心思想】
//    冲四 (rush4)：落子后形成四连且只有一端开放，
//    对手唯一的正确应对是堵住那个唯一开放端。
//    因此防守方分支因子 = 1，搜索树近似线性，深度可达 20+。
//
//  【Bug 修复：正确处理双冲四】
//    双冲四 = 落子后棋面上攻击方有 ≥2 个立即赢棋点。
//    对手只能堵一个，必输。用 countWinMoves 计数赢棋点判断。
//
//  【时间限制】
//    每次进入 VCF 时检查 elapsed()，防止深层搜索超时。
// ═══════════════════════════════════════════════════════════════

// 统计当前棋面 color 方的立即赢棋点数（最多数到 2 即返回）
static int countWinMoves(int color, int &wx, int &wy) {
    int cnt=0; wx=-1; wy=-1;
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++) {
            if (board[i][j]!=EMPTY) continue;
            board[i][j]=color;
            bool w=checkWin(i,j,color);
            board[i][j]=EMPTY;
            if (w) {
                cnt++;
                if (wx<0) { wx=i; wy=j; }
                if (cnt>=2) return 2;
            }
        }
    return cnt;
}

// VCF 时间截止（避免深层搜索超时）
static double vcf_deadline;

static bool vcfDFS(int attacker, int defender, int depth,
                   int &firstX, int &firstY, bool isRoot) {
    // 时间检查：VCF 不能占用太多时间
    if (elapsed() > vcf_deadline) return false;
    if (depth <= 0) return false;

    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++) {
            if (board[i][j]!=EMPTY || !hasNeighbor(i,j)) continue;

            board[i][j] = attacker;

            // 直接连五
            if (checkWin(i, j, attacker)) {
                if (isRoot) { firstX=i; firstY=j; }
                board[i][j] = EMPTY;
                return true;
            }

            // 统计落子后攻击方的赢棋点数：
            //   ≥2 → 双冲四/活四组合，对手无法同时堵 → 必胜
            //   =1 → 单冲四，找对手唯一应答并递归
            //   =0 → 此落子不形成 VCF 攻击，跳过
            int wx=-1, wy=-1;
            int wc = countWinMoves(attacker, wx, wy);

            if (wc >= 2) {
                if (isRoot) { firstX=i; firstY=j; }
                board[i][j] = EMPTY;
                return true;
            }

            if (wc == 1) {
                // 对手唯一应答：堵住 (wx,wy)
                board[wx][wy] = defender;
                bool ok = vcfDFS(attacker, defender, depth-1, firstX, firstY, false);
                board[wx][wy] = EMPTY;
                if (ok) {
                    if (isRoot) { firstX=i; firstY=j; }
                    board[i][j] = EMPTY;
                    return true;
                }
            }

            board[i][j] = EMPTY;
        }
    return false;
}

// VCF 对外接口：返回 1=我方必胜(wx,wy 为第一步), -1=需防对方VCF, 0=无
static int vcfCheck(int &wx, int &wy) {
    vcf_deadline = min(elapsed() + 0.08, TL * 0.35); // 最多用 0.08s 或 35% 预算

    if (vcfDFS(P1, P2, 20, wx, wy, true)) return 1;

    vcf_deadline = min(elapsed() + 0.05, TL * 0.40);
    int d1=-1, d2=-1;
    if (vcfDFS(P2, P1, 12, d1, d2, true)) {
        if (getWinMove(P2, wx, wy)) return -1;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════
//  §4  Zobrist 哈希 + 置换表
//
//  【论文】Zobrist, A.L. (1970). "A New Hashing Method with Application
//          for Game Playing." Tech Report 88, Univ. of Wisconsin.
//
//  增量更新：每次落子/撤子只需 O(1) 的 XOR 操作。
//  置换表：存储已搜索节点避免重复计算。
// ═══════════════════════════════════════════════════════════════

static uint64_t ZT[SIZE][SIZE][3]; // ZT[x][y][color]
static uint64_t curHash = 0;

struct TTEntry {
    uint64_t hash;
    int      score, depth;
    uint8_t  flag;    // 0=精确, 1=下界, 2=上界
    int8_t   bx, by; // 最优走法
};
static const int TT_SIZE = (1<<18); // ~262K 条目，约 5MB
static TTEntry TT[TT_SIZE];

static void initZobrist() {
    mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++)
            for (int c=1;c<=2;c++)
                ZT[i][j][c]=rng();
}

inline void placeAB(int x,int y,int c) { board[x][y]=c; curHash^=ZT[x][y][c]; }
inline void removeAB(int x,int y,int c) { board[x][y]=EMPTY; curHash^=ZT[x][y][c]; }

static void computeHash() {
    curHash=0;
    for (int i=0;i<SIZE;i++)
        for (int j=0;j<SIZE;j++)
            if (board[i][j]) curHash^=ZT[i][j][board[i][j]];
}

// ═══════════════════════════════════════════════════════════════
//  §5  历史启发 + Killer Move
//
//  【论文】Schaeffer, J. (1989). "The History Heuristic and Alpha-Beta
//          Search Enhancements in Practice."
//          IEEE Trans. Pattern Anal. Mach. Intell., 11(11), 1203-1212.
// ═══════════════════════════════════════════════════════════════

static int histScore[SIZE][SIZE];
static int8_t killerX[20][2], killerY[20][2];

static void addKiller(int d, int x, int y) {
    if (killerX[d][0]==x && killerY[d][0]==y) return;
    killerX[d][1]=killerX[d][0]; killerY[d][1]=killerY[d][0];
    killerX[d][0]=(int8_t)x;     killerY[d][0]=(int8_t)y;
}

static void clearABState() {
    memset(TT,       0, sizeof(TT));
    memset(histScore,0, sizeof(histScore));
    memset(killerX, -1, sizeof(killerX));
    memset(killerY, -1, sizeof(killerY));
}

// ═══════════════════════════════════════════════════════════════
//  §6  迭代加深 Alpha-Beta
//
//  【迭代加深】Korf, R.E. (1985). Artif. Intell. 27(1), 97-109.
// ═══════════════════════════════════════════════════════════════

static vector<Move> genMovesAB(int color, int abD) {
    // 置换表最优走法
    int ttbx=-1, ttby=-1;
    TTEntry &te=TT[curHash&(TT_SIZE-1)];
    if (te.hash==curHash && te.bx>=0) { ttbx=te.bx; ttby=te.by; }

    auto mv=genMoves(color, histScore, 20);

    for (int i=0;i<(int)mv.size();i++)
        if (mv[i].x==ttbx && mv[i].y==ttby) { swap(mv[i],mv[0]); break; }
    for (int slot=0;slot<2;slot++) {
        int kx=killerX[abD][slot], ky=killerY[abD][slot];
        if (kx<0) continue;
        for (int i=1;i<(int)mv.size();i++)
            if (mv[i].x==kx && mv[i].y==ky) { swap(mv[i],mv[1]); break; }
    }
    return mv;
}

static int alphaBeta(int color, int depth, int alpha, int beta, int abD) {
    int idx=(int)(curHash&(TT_SIZE-1));
    TTEntry &te=TT[idx];
    if (te.hash==curHash && te.depth>=depth) {
        if (te.flag==0) return te.score;
        if (te.flag==1 && te.score>=beta)  return te.score;
        if (te.flag==2 && te.score<=alpha) return te.score;
    }

    if (depth==0) {
        int s=evaluateBoard();
        TT[idx]={curHash,s,0,0,-1,-1};
        return s;
    }

    auto mv=genMovesAB(color,abD);
    if (mv.empty()) {
        int s=evaluateBoard();
        TT[idx]={curHash,s,depth,0,-1,-1};
        return s;
    }

    bool isMax=(color==P1);
    int best=isMax?-INF:INF;
    int8_t bx=-1, by=-1;

    for (auto &m:mv) {
        placeAB(m.x,m.y,color);
        if (checkWin(m.x,m.y,color)) {
            removeAB(m.x,m.y,color);
            int s=isMax?(SC_FIVE+depth):-(SC_FIVE+depth);
            TT[idx]={curHash,s,depth,0,(int8_t)m.x,(int8_t)m.y};
            return s;
        }
        int s=alphaBeta(3-color,depth-1,alpha,beta,abD+1);
        removeAB(m.x,m.y,color);

        if (isMax) {
            if (s>best) { best=s; bx=m.x; by=m.y; }
            alpha=max(alpha,s);
        } else {
            if (s<best) { best=s; bx=m.x; by=m.y; }
            beta=min(beta,s);
        }
        if (alpha>=beta) {
            histScore[m.x][m.y]+=depth*depth;
            addKiller(abD,m.x,m.y);
            TT[idx]={curHash,best,depth,(uint8_t)(isMax?1:2),bx,by};
            return best;
        }
    }
    TT[idx]={curHash,best,depth,0,bx,by};
    return best;
}

// 迭代加深入口，返回最佳走法 (x,y)
static pair<int,int> iterDeepAB() {
    computeHash();
    clearABState();

    // 候选走法列表（默认走法 = 第一候选）
    auto cands=genMoves(P1,nullptr,20);
    if (cands.empty()) {
        // 极端兜底（棋盘接近全满）
        for (int i=0;i<SIZE;i++)
            for (int j=0;j<SIZE;j++)
                if (board[i][j]==EMPTY) return {i,j};
        return {0,0};
    }
    pair<int,int> bestMove={cands[0].x, cands[0].y};

    for (int depth=2; depth<=8; depth+=2) {
        if (elapsed()>TL*0.45) break;
        int alpha=-INF, beta=INF, bestScore=-INF;
        for (auto &m:cands) {
            if (elapsed()>TL*0.50) break;
            placeAB(m.x,m.y,P1);
            if (checkWin(m.x,m.y,P1)) { removeAB(m.x,m.y,P1); return {m.x,m.y}; }
            int s=alphaBeta(P2,depth-1,alpha,beta,1);
            removeAB(m.x,m.y,P1);
            if (s>bestScore) { bestScore=s; bestMove={m.x,m.y}; }
            alpha=max(alpha,s);
        }
    }
    return bestMove;
}

// ═══════════════════════════════════════════════════════════════
//  §7  MCTS —— 落子/撤销方案 + MCTS-Solver
//
//  【基础】Kocsis & Szepesvari (2006). ECML. UCB1 公式。
//  【Solver】Winands, Bjornsson, Saito (2008). CG.
//
//  【关键修复：使用落子/撤销方案（参照 v5）】
//    v6 初版使用 replayPath 每次从根快照重放，存在两个问题：
//    1. 函数结束时棋盘未还原，导致后续安全检查失效 → INVALID INPUT
//    2. 逻辑复杂，边界情况难以处理
//    修复：完全改用落子/撤销方案：
//    - Selection 阶段：落子进入子节点
//    - Backpropagation 后：沿父链撤销所有落子
//    - 棋盘状态始终与迭代同步，循环结束时自动恢复原状
//
//  【UCB1 修正（v5 Bug 修复）】
//    正确公式：exploit = wins/visits（child.wins = 到达该节点落子方的胜数）
// ═══════════════════════════════════════════════════════════════

enum NodeState : int8_t { NS_UNKNOWN=0, NS_WIN=1, NS_LOSS=2 };

static const int MAX_CANDS = 28;

// 紧凑型节点（不含棋面快照，约 190 字节/节点）
struct MCTSNode {
    int8_t    mx, my;              // 到达此节点的走法
    int8_t    pjm;                 // playerJustMoved
    NodeState state;               // Solver 状态
    int       visits, wins;
    int32_t   parent;
    int32_t   children[MAX_CANDS];
    int8_t    nChildren;
    int8_t    untried[MAX_CANDS][2];
    int8_t    nUntried;
};
// 50000 节点 × 190 字节 ≈ 9.5MB

static const int POOL = 50000;
static MCTSNode pool[POOL];
static int poolTop = 0;

static int newNode(int par, int8_t pmx, int8_t pmy, int8_t pjm_val,
                   const vector<Move> &cands) {
    if (poolTop>=POOL) return -1;
    MCTSNode &n=pool[poolTop];
    n.mx=pmx; n.my=pmy; n.pjm=pjm_val;
    n.state=NS_UNKNOWN; n.visits=0; n.wins=0;
    n.parent=par; n.nChildren=0; n.nUntried=0;
    for (auto &m:cands) {
        if (n.nUntried>=MAX_CANDS) break;
        n.untried[n.nUntried][0]=(int8_t)m.x;
        n.untried[n.nUntried][1]=(int8_t)m.y;
        n.nUntried++;
    }
    return poolTop++;
}

// UCB1 选择（修正后的 wins/visits）
static int selectChild(int ni) {
    MCTSNode &n=pool[ni];
    double lnN=log((double)max(1,n.visits));
    const double C=1.414;
    int best=-1; double bval=-1e18;
    for (int i=0;i<n.nChildren;i++) {
        int ci=n.children[i]; if (ci<0) continue;
        MCTSNode &c=pool[ci];
        if (c.state==NS_WIN)  return ci;
        if (c.state==NS_LOSS) continue;
        if (c.visits==0) return ci;
        double v=(double)c.wins/c.visits + C*sqrt(lnN/c.visits);
        if (v>bval) { bval=v; best=ci; }
    }
    return best;
}

// MCTS-Solver 状态传播
static void solverUpdate(int ni) {
    MCTSNode &n=pool[ni];
    bool anyWin=false, allLoss=true;
    for (int i=0;i<n.nChildren;i++) {
        int ci=n.children[i]; if (ci<0) continue;
        if (pool[ci].state==NS_WIN)  anyWin=true;
        if (pool[ci].state!=NS_LOSS) allLoss=false;
    }
    if (anyWin) { n.state=NS_LOSS; return; }
    if (allLoss && n.nUntried==0 && n.nChildren>0) n.state=NS_WIN;
}

// 快速模拟（优先赢棋 > 堵对手 > 评分采样）
// 【重要】模拟过程中修改 board，但调用方已在外层保存/恢复
static int smartSim(int sc) {
    for (int step=0;step<40;step++) {
        int wx,wy;
        if (getWinMove(sc,wx,wy)) { board[wx][wy]=sc; return sc; }
        int opp=3-sc;
        if (getWinMove(opp,wx,wy)) { board[wx][wy]=sc; sc=opp; continue; }
        auto mv=genMoves(sc,nullptr,8);
        if (mv.empty()) break;
        int pick=rand()%min(5,(int)mv.size());
        board[mv[pick].x][mv[pick].y]=sc;
        sc=opp;
    }
    return 0;
}

// MCTS 主搜索（落子/撤销方案，棋盘状态自动还原）
static pair<int,int> mctsSearch(double deadline) {
    poolTop=0;

    // 根节点：pjm=P2（上一手是对方落子），下一步该 P1 走
    auto rootCands=genMoves(P1,nullptr,MAX_CANDS);
    if (rootCands.empty()) return {-1,-1};
    int root=newNode(-1,-1,-1,(int8_t)P2,rootCands);

    while (elapsed()<deadline && poolTop<POOL-2) {

        // ══ 阶段1：Selection ══
        int node=root;
        int simPlayer=P1; // 从根出发，P1 先走

        while (pool[node].nUntried==0 && pool[node].nChildren>0
               && pool[node].state==NS_UNKNOWN) {
            int next=selectChild(node);
            if (next<0) break;
            // 落子进入子节点
            board[(int)(uint8_t)pool[next].mx]
                 [(int)(uint8_t)pool[next].my] = simPlayer;
            simPlayer=3-simPlayer;
            node=next;
        }

        // 确定性节点直接传播
        if (pool[node].state != NS_UNKNOWN) {
            int result=(pool[node].state==NS_WIN)?(int)pool[node].pjm:0;
            int cur=node;
            while (cur>=0) {
                pool[cur].visits++;
                if (pool[cur].pjm==result) pool[cur].wins++;
                solverUpdate(cur);
                if (cur==root) break;
                // 撤销本节点落子
                board[(int)(uint8_t)pool[cur].mx]
                     [(int)(uint8_t)pool[cur].my]=EMPTY;
                cur=pool[cur].parent;
            }
            continue;
        }

        // ══ 阶段2：Expansion ══
        int simResult=0;
        if (pool[node].nUntried>0) {
            int idx=rand()%pool[node].nUntried;
            int8_t ux=pool[node].untried[idx][0];
            int8_t uy=pool[node].untried[idx][1];
            pool[node].untried[idx][0]=pool[node].untried[pool[node].nUntried-1][0];
            pool[node].untried[idx][1]=pool[node].untried[pool[node].nUntried-1][1];
            pool[node].nUntried--;

            // 落子（simPlayer 是当前要走的颜色）
            board[(int)(uint8_t)ux][(int)(uint8_t)uy]=simPlayer;
            int nextPjm=simPlayer;

            if (checkWin((int)(uint8_t)ux,(int)(uint8_t)uy,simPlayer)) {
                vector<Move> ec;
                int child=newNode(node,ux,uy,(int8_t)nextPjm,ec);
                if (child>=0) {
                    pool[node].children[pool[node].nChildren++]=child;
                    pool[child].state=NS_WIN;
                    node=child;
                }
                simResult=simPlayer;
            } else {
                int childNext=3-simPlayer;
                auto cc=genMoves(childNext,nullptr,MAX_CANDS);
                int child=newNode(node,ux,uy,(int8_t)nextPjm,cc);
                if (child>=0) {
                    pool[node].children[pool[node].nChildren++]=child;
                    node=child;
                    // ══ 阶段3：Simulation ══
                    // 保存并还原：模拟在当前棋面基础上随机走棋
                    int bkp[SIZE][SIZE];
                    memcpy(bkp,board,sizeof(board));
                    simResult=smartSim(childNext);
                    memcpy(board,bkp,sizeof(board));
                }
            }
        }

        // ══ 阶段4：Backpropagation + 撤销落子 ══
        // 从当前节点向上传播，同时撤销 Selection 阶段落下的所有棋子
        int cur=node;
        while (cur>=0) {
            pool[cur].visits++;
            if (pool[cur].pjm==simResult) pool[cur].wins++;
            solverUpdate(cur);
            if (cur==root) break;
            // 撤销这一步的落子（恢复棋盘）
            board[(int)(uint8_t)pool[cur].mx]
                 [(int)(uint8_t)pool[cur].my]=EMPTY;
            cur=pool[cur].parent;
        }
        // ─── 循环结束时，board 已还原到 MCTS 开始前的状态 ───
    }

    // 选最优根子节点（确定性必赢 > 访问次数最多）
    MCTSNode &r=pool[root];
    int best=-1, bVis=0;
    for (int i=0;i<r.nChildren;i++) {
        int ci=r.children[i]; if (ci<0) continue;
        if (pool[ci].state==NS_WIN) return {(int)(uint8_t)pool[ci].mx,
                                            (int)(uint8_t)pool[ci].my};
        if (pool[ci].visits>bVis) { bVis=pool[ci].visits; best=ci; }
    }
    if (best>=0) return {(int)(uint8_t)pool[best].mx,
                         (int)(uint8_t)pool[best].my};
    return {-1,-1};
}

// ═══════════════════════════════════════════════════════════════
//  §8  主函数 —— 与 Gomoku_simple.cpp 完全一致的 I/O 格式
// ═══════════════════════════════════════════════════════════════
int main() {
    g_start = clock();
    srand((unsigned)time(nullptr) ^ (unsigned)clock());
    memset(board, 0, sizeof(board));
    initZobrist();

    int n, x, y;
    cin >> n;

    // 读取 n-1 回合历史（与 Gomoku_simple.cpp 完全一致）
    // board[x][y]=P2 → 对方棋子；board[x][y]=P1 → 本方棋子
    for (int i=0; i<n-1; i++) {
        cin >> x >> y; if (x != -1) board[x][y] = P2;  // 对方
        cin >> x >> y; if (x != -1) board[x][y] = P1;  // 本方
    }
    // 第 n 回合对方最新落子
    cin >> x >> y;
    if (x != -1) board[x][y] = P2;

    int new_x = -1, new_y = -1;

    // ─── 第 1 回合特殊处理（开局/换手）────────────────────────
    if (n == 1) {
        if (x == -1) {
            // 我方先手：走天元
            new_x = SIZE/2; new_y = SIZE/2;
        } else {
            // 我方后手：判断是否换手
            int dist = abs(x - SIZE/2) + abs(y - SIZE/2);
            // 对方走在中心附近（曼哈顿距离 ≤ 2）→ 换手
            if (dist <= 2) { new_x = -1; new_y = -1; }
            else           { new_x = SIZE/2; new_y = SIZE/2; }
        }
        printf("%d %d\n", new_x, new_y);
        return 0;
    }

    // ─── n ≥ 2：5 层决策漏斗 ─────────────────────────────────

    // 层1：即时胜/防
    {
        int wx, wy;
        if (getWinMove(P1, wx, wy)) { new_x=wx; new_y=wy; goto OUTPUT; }
        if (getWinMove(P2, wx, wy)) { new_x=wx; new_y=wy; goto OUTPUT; }
    }

    // 层2：双威胁战术（活四/双冲四组合必杀）
    {
        auto cands = genMoves(P1, nullptr, 20);
        for (auto &m : cands) {
            int s = analyzePoint(m.x, m.y, P1);
            if (s >= SC_L4 || s >= 2*SC_R4) {
                new_x=m.x; new_y=m.y; goto OUTPUT;
            }
        }
    }

    // 层3：VCF 必杀搜索（带时间限制，论文：Allis 1993）
    {
        int wx=-1, wy=-1;
        if (vcfCheck(wx, wy) != 0) { new_x=wx; new_y=wy; goto OUTPUT; }
    }

    // 层4：迭代加深 Alpha-Beta（Zobrist TT + 历史启发）
    {
        auto [abx, aby] = iterDeepAB();
        new_x = abx; new_y = aby; // 先用 AB 结果保底
    }

    // 层5：MCTS 精搜索（剩余时间，MCTS-Solver 增强）
    if (elapsed() < TL * 0.80) {
        auto [mx, my] = mctsSearch(TL * 0.92);
        // 只有当 MCTS 返回合法空格时才采用
        if (mx >= 0 && mx < SIZE && my >= 0 && my < SIZE
            && board[mx][my] == EMPTY) {
            new_x = mx; new_y = my;
        }
    }

OUTPUT:
    // 安全兜底：确保输出位置合法（不越界、不重复）
    if (new_x<0 || new_x>=SIZE || new_y<0 || new_y>=SIZE
        || board[new_x][new_y] != EMPTY) {
        // 找一个有邻子的空格
        for (int i=0; i<SIZE; i++)
            for (int j=0; j<SIZE; j++)
                if (board[i][j]==EMPTY && hasNeighbor(i,j)) {
                    new_x=i; new_y=j; goto DONE;
                }
        // 极端兜底（棋盘几乎全满）
        for (int i=0; i<SIZE; i++)
            for (int j=0; j<SIZE; j++)
                if (board[i][j]==EMPTY) { new_x=i; new_y=j; goto DONE; }
    }

DONE:
    printf("%d %d\n", new_x, new_y);
    return 0;
}
