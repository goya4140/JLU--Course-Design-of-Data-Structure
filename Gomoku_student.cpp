/*
 * ============================================================
 *  五子棋 AI —— 课程设计版（学生答辩版）
 *  平台：Botzone (Gomoku-Swap1 规则)
 *  编译：g++ -O2 -std=c++11 -o gomoku Gomoku_student.cpp
 *
 *  【整体设计思路：5 层决策漏斗】
 *
 *  每一步棋，程序依次尝试以下 5 层策略，
 *  只要某层找到答案就立刻输出，不进入下一层：
 *
 *   层 1  即时胜负判断   → 我能赢就赢，对手能赢就堵
 *   层 2  双威胁检测     → 形成"两路同时致命"的必胜局面
 *   层 3  VCF 必杀搜索   → 连续冲四逼对手，寻找必胜序列
 *   层 4  Alpha-Beta 搜索→ 博弈树深度搜索 + 置换表加速
 *   层 5  MCTS 精化     → 蒙特卡洛树搜索，利用剩余时间优化
 *
 *  【棋盘编码规则（与样例程序完全一致）】
 *    board[x][y] = 0  → 空格
 *    board[x][y] = 1  → 我方棋子 (ME)
 *    board[x][y] = 2  → 对方棋子 (OPP)
 *
 *  【重要：时间控制】
 *    Botzone 每步限时 1 秒，程序内部预算 0.90 秒
 *    在耗时循环中必须检查时间，防止超时判负
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <random>    // 用于 mt19937 生成高质量随机数（Zobrist 哈希初始化）
using namespace std;

// ─────────────────────────────────────────────────────────────
// 第 0 节：全局常量与基础工具
// ─────────────────────────────────────────────────────────────

const int SIZE  = 15;   // 棋盘尺寸（15×15）
const int EMPTY = 0;    // 空格标记
const int ME    = 1;    // 我方棋子标记
const int OPP   = 2;    // 对方棋子标记
const int INF   = 100000000;  // 搜索用无穷大

// 4 个方向向量：水平 / 垂直 / 右斜 / 左斜
// 棋盘上的连子检测只需检查这 4 个方向（双向合计 8 条线）
const int DX[4] = {0, 1, 1,  1};
const int DY[4] = {1, 0, 1, -1};

// 棋盘全局数组（程序启动后 memset 清零）
int board[SIZE][SIZE];

// ── 计时工具 ──
clock_t g_startTime;  // 程序启动时间（在 main 最开始记录）

// 返回程序运行至今的秒数
double getElapsed() {
    return (double)(clock() - g_startTime) / CLOCKS_PER_SEC;
}

// 总时间预算：0.90 秒（Botzone 限 1 秒，留 0.10 秒余量防止误判）
const double TIME_LIMIT = 0.90;

// ─────────────────────────────────────────────────────────────
// 第 1 节：棋形评估
// ─────────────────────────────────────────────────────────────
/*
 * 【算法思想】
 *   五子棋的"棋形"指某种连子格局，不同棋形威胁程度不同。
 *   常见棋形（从强到弱）：
 *     连五   ── 已赢，分值最高
 *     活四   ── 两端均开放的四连，下一步必赢
 *     冲四   ── 一端开放的四连，对手必须堵
 *     活三   ── 两端开放的三连，威胁较大
 *     眠三   ── 一端开放的三连
 *     活二 / 眠二
 *
 *   评估函数 = 我方所有棋形分之和 − 对方所有棋形分之和
 *   分值为正 → 我方占优；分值为负 → 对方占优
 */

// 棋形分值（数字大小体现威胁程度，仅需相对大小正确）
const int SCORE_FIVE = 1000000;  // 连五（必赢）
const int SCORE_L4   =   50000;  // 活四
const int SCORE_R4   =   10000;  // 冲四
const int SCORE_L3   =    5000;  // 活三
const int SCORE_R3   =     500;  // 眠三
const int SCORE_L2   =     200;  // 活二
const int SCORE_R2   =      20;  // 眠二

/*
 * 单方向评分
 * 假设已在 board[x][y] 落下 color 子，
 * 沿方向 (dx, dy) 向两端延伸，统计：
 *   cnt      → 连续棋子数（含 (x,y) 本身）
 *   openLeft / openRight → 左/右端是否为空格（开放）
 * 根据 cnt 和开放端数，对照棋形表返回得分。
 */
int scoreOneDir(int x, int y, int dx, int dy, int color) {
    int cnt = 1;         // 当前已有 (x,y) 这一颗
    int openLeft  = 0;   // 左端（反方向）是否开放
    int openRight = 0;   // 右端（正方向）是否开放

    // 向正方向（dx, dy）延伸
    for (int k = 1; k <= 4; k++) {
        int nx = x + k * dx;
        int ny = y + k * dy;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) break;  // 越界停止
        if (board[nx][ny] == color) {
            cnt++;  // 同色，继续延伸
        } else {
            if (board[nx][ny] == EMPTY) openRight = 1;  // 空格 → 此端开放
            break;  // 遇到异色棋子或边界 → 不能再延伸
        }
    }
    // 向反方向（-dx, -dy）延伸
    for (int k = 1; k <= 4; k++) {
        int nx = x - k * dx;
        int ny = y - k * dy;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) break;
        if (board[nx][ny] == color) {
            cnt++;
        } else {
            if (board[nx][ny] == EMPTY) openLeft = 1;
            break;
        }
    }

    int openEnds = openLeft + openRight;  // 0=两端封闭, 1=单端开放, 2=两端开放

    // 对照棋形表打分
    if (cnt >= 5) return SCORE_FIVE;
    if (cnt == 4) return openEnds == 2 ? SCORE_L4 : (openEnds == 1 ? SCORE_R4 : 0);
    if (cnt == 3) return openEnds == 2 ? SCORE_L3 : (openEnds == 1 ? SCORE_R3 : 0);
    if (cnt == 2) return openEnds == 2 ? SCORE_L2 : (openEnds == 1 ? SCORE_R2 : 0);
    return 0;
}

/*
 * 综合 4 个方向，计算在 (x,y) 落 color 子后的总威胁分
 * （落子前 board[x][y] 应为 EMPTY；调用时不实际改变棋盘）
 */
int evalPoint(int x, int y, int color) {
    int total = 0;
    for (int d = 0; d < 4; d++)
        total += scoreOneDir(x, y, DX[d], DY[d], color);
    return total;
}

/*
 * 全局棋面静态估值（以 ME 视角，正值对我有利）
 * 遍历棋盘上所有棋子，分别累加双方得分
 */
int evalBoard() {
    int myScore  = 0;
    int oppScore = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] == ME)  myScore  += evalPoint(i, j, ME);
            if (board[i][j] == OPP) oppScore += evalPoint(i, j, OPP);
        }
    }
    return myScore - oppScore;
}

// ─────────────────────────────────────────────────────────────
// 第 2 节：基础工具函数
// ─────────────────────────────────────────────────────────────

/*
 * 胜负检测
 * 判断在 (x,y) 落 color 子后，是否已形成连五。
 * 原理：沿 4 个方向各统计连续同色棋子数，任一方向 ≥5 则获胜。
 */
bool checkWin(int x, int y, int color) {
    for (int d = 0; d < 4; d++) {
        int cnt = 1;
        // 正方向延伸
        int nx = x + DX[d], ny = y + DY[d];
        while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[nx][ny] == color) {
            cnt++;
            nx += DX[d]; ny += DY[d];
        }
        // 反方向延伸
        nx = x - DX[d]; ny = y - DY[d];
        while (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[nx][ny] == color) {
            cnt++;
            nx -= DX[d]; ny -= DY[d];
        }
        if (cnt >= 5) return true;
    }
    return false;
}

/*
 * 邻近性剪枝（Proximity Pruning）
 * 判断 (x,y) 是否在已有棋子的 radius 格以内。
 *
 * 【为什么需要这个函数？】
 *   15×15 棋盘有 225 格，但实际有意义的候选落点只在棋子附近。
 *   通过 hasNeighbor 过滤，可以把候选走法从 200+ 个减少到 20~30 个，
 *   大幅降低搜索树的宽度，提高搜索速度。
 */
bool hasNeighbor(int x, int y, int radius = 2) {
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE)
                if (board[nx][ny] != EMPTY) return true;
        }
    }
    return false;
}

/*
 * 寻找立即赢棋点
 * 扫描棋盘，找一个空格，落 color 子后能立即连五。
 * 找到则将坐标写入 (wx, wy) 并返回 true。
 */
bool getWinMove(int color, int &wx, int &wy) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != EMPTY) continue;
            board[i][j] = color;        // 假设落子
            bool win = checkWin(i, j, color);
            board[i][j] = EMPTY;        // 恢复
            if (win) { wx = i; wy = j; return true; }
        }
    }
    return false;
}

// ─────────────────────────────────────────────────────────────
// 第 3 节：候选走法生成
// ─────────────────────────────────────────────────────────────

// 走法结构体（坐标 + 评估分）
struct Move {
    int x, y, score;
};

/*
 * 生成带评分的候选走法列表
 *
 * 【生成规则】
 *   1. 只考虑有邻近棋子的空格（hasNeighbor 剪枝）
 *   2. 评分 = 落点对我方的价值 + 落点对对方的价值（攻守兼顾）
 *   3. 如果有历史启发分（histScore），额外加权（棋形搜索中常用棋步优先）
 *   4. 按分值降序排列，取前 topN 个
 *
 * 【为什么降序排列很重要？】
 *   Alpha-Beta 剪枝的效率依赖于走法排序：好的走法越早搜索，
 *   越早产生强剪枝，搜索树规模指数级减小。
 *
 * @param color     当前落子方
 * @param histScore 历史启发分表（为 nullptr 时不使用）
 * @param topN      最多返回多少个候选走法
 */
vector<Move> genMoves(int color, int histScore[][SIZE] = nullptr, int topN = 20) {
    int opp = 3 - color;  // 对手颜色（1↔2 互换）
    vector<Move> moves;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != EMPTY || !hasNeighbor(i, j)) continue;
            int s = evalPoint(i, j, color) + evalPoint(i, j, opp);
            if (histScore != nullptr) s += histScore[i][j] * 8;  // 历史启发加权
            moves.push_back({i, j, s});
        }
    }

    // 按分值从高到低排序
    sort(moves.begin(), moves.end(), [](const Move &a, const Move &b) {
        return a.score > b.score;
    });

    if ((int)moves.size() > topN) moves.resize(topN);
    return moves;
}

// ─────────────────────────────────────────────────────────────
// 第 4 节：VCF 连续冲四必杀搜索
// ─────────────────────────────────────────────────────────────
/*
 * 【VCF 是什么？】
 *   VCF = Victory by Continuous Fours（连续冲四获胜）
 *
 * 【核心思想】
 *   "冲四"指落子后形成四连且只有一端开放，对手必须堵住那个唯一开口。
 *   因此防守方的分支因子 ≈ 1（几乎没有选择），搜索树接近一条链，
 *   可以搜到很深（20+ 步）而不会超时。
 *
 *   如果攻击方连续冲四，最终形成"五连"或"双冲四"局面，
 *   就找到了一个无论对手如何应对都必赢的序列，称为 VCF 必杀。
 *
 * 【双冲四的处理】
 *   落子后若攻击方同时有 ≥2 个赢棋点，对手只能堵一个，必输。
 *   通过 countWinPoints 函数检测此情况。
 *
 * 【时间限制】
 *   VCF 搜索有自己的时间截止点，防止在深层搜索中超时。
 */

double vcfDeadline;  // VCF 搜索截止时间（秒）

/*
 * 统计 color 方当前有多少个立即赢棋点
 * 最多统计到 2（找到 2 个就够，不再继续）
 * 找到的第一个赢棋点写入 (wx, wy)
 */
int countWinPoints(int color, int &wx, int &wy) {
    int cnt = 0;
    wx = -1; wy = -1;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != EMPTY) continue;
            board[i][j] = color;
            bool win = checkWin(i, j, color);
            board[i][j] = EMPTY;
            if (win) {
                cnt++;
                if (wx < 0) { wx = i; wy = j; }  // 记录第一个赢棋点
                if (cnt >= 2) return 2;             // 找到 2 个就够了
            }
        }
    }
    return cnt;
}

/*
 * VCF 深度优先搜索
 *
 * @param attacker  进攻方
 * @param defender  防守方
 * @param depth     剩余搜索深度（每递归一层减 1）
 * @param firstX/Y  记录必杀序列的第一步坐标
 * @param isRoot    是否是搜索的根层（只有根层才更新 firstX/Y）
 * @return true 表示找到必杀序列
 */
bool vcfDFS(int attacker, int defender, int depth,
            int &firstX, int &firstY, bool isRoot) {
    // 时间检查：VCF 不能无限占用时间
    if (getElapsed() > vcfDeadline) return false;
    if (depth <= 0) return false;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != EMPTY || !hasNeighbor(i, j)) continue;

            board[i][j] = attacker;  // 试探：攻击方落子

            // 情况 1：直接连五 → 找到必杀
            if (checkWin(i, j, attacker)) {
                if (isRoot) { firstX = i; firstY = j; }
                board[i][j] = EMPTY;
                return true;
            }

            // 检测落子后攻击方的赢棋点数量
            int wx = -1, wy = -1;
            int winCount = countWinPoints(attacker, wx, wy);

            // 情况 2：双冲四/活四组合 → 对手无法同时堵住 → 必胜
            if (winCount >= 2) {
                if (isRoot) { firstX = i; firstY = j; }
                board[i][j] = EMPTY;
                return true;
            }

            // 情况 3：单冲四 → 对手唯一应答是堵住 (wx, wy)，递归继续
            if (winCount == 1) {
                board[wx][wy] = defender;  // 对手被迫应对
                bool found = vcfDFS(attacker, defender, depth - 1, firstX, firstY, false);
                board[wx][wy] = EMPTY;     // 撤销对手落子
                if (found) {
                    if (isRoot) { firstX = i; firstY = j; }
                    board[i][j] = EMPTY;
                    return true;
                }
            }

            board[i][j] = EMPTY;  // 撤销攻击方落子（此路不通）
        }
    }
    return false;
}

/*
 * VCF 对外接口
 * 返回值：
 *   1  → 我方存在 VCF 必杀，(wx,wy) 为第一步
 *  -1  → 对方存在 VCF 威胁，(wx,wy) 为防守位置
 *   0  → 没有 VCF 相关局面
 */
int vcfCheck(int &wx, int &wy) {
    // 先搜索我方是否有必杀（最多用 0.08 秒）
    vcfDeadline = min(getElapsed() + 0.08, TIME_LIMIT * 0.35);
    if (vcfDFS(ME, OPP, 20, wx, wy, true)) return 1;

    // 再检测对方是否有 VCF 威胁（最多再用 0.05 秒）
    int dx = -1, dy = -1;
    vcfDeadline = min(getElapsed() + 0.05, TIME_LIMIT * 0.40);
    if (vcfDFS(OPP, ME, 12, dx, dy, true)) {
        // 对方有 VCF，优先堵住对方立即赢棋点
        if (getWinMove(OPP, wx, wy)) return -1;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────
// 第 5 节：Zobrist 哈希 + 置换表（Alpha-Beta 加速）
// ─────────────────────────────────────────────────────────────
/*
 * 【Zobrist 哈希是什么？】
 *   每个棋盘位置、每种棋子颜色，预先分配一个随机的 64 位整数。
 *   棋面哈希值 = 所有棋子对应随机数的 XOR。
 *
 *   关键性质：
 *   - 落一颗棋子：currentHash ^= ZT[x][y][color]   （O(1)）
 *   - 撤一颗棋子：currentHash ^= ZT[x][y][color]   （XOR 两次等于没做，自动恢复）
 *
 * 【置换表（Transposition Table）是什么？】
 *   博弈树中，不同走棋顺序可能到达同一棋面（置换现象）。
 *   置换表用哈希值作为 key，缓存已搜索棋面的结果，避免重复计算。
 *   这是 Alpha-Beta 搜索最重要的加速手段之一。
 */

// Zobrist 随机数表：ZT[位置x][位置y][棋子颜色]
unsigned long long ZT[SIZE][SIZE][3];

// 当前棋面的哈希值（随落子/撤子实时更新）
unsigned long long currentHash = 0;

// 置换表条目结构
struct TTEntry {
    unsigned long long hash;  // 棋面哈希（用于验证，防止哈希碰撞误用）
    int score;                // 该棋面的搜索结果分值
    int depth;                // 搜索时使用的深度（越大越精确）
    int flag;                 // 分值类型：0=精确值, 1=下界(β截断), 2=上界(α截断)
    int bestX, bestY;         // 该棋面的最优走法（-1 表示无记录）
};

const int TT_SIZE = (1 << 18);  // 262144 个条目，约 10MB
TTEntry TT[TT_SIZE];

/*
 * 初始化 Zobrist 随机数表
 * 使用 mt19937 生成高质量随机数（固定种子保证可重复性）
 */
void initZobrist() {
    mt19937 rng(42);  // 固定种子，每次运行结果相同
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int c = 1; c <= 2; c++) {
                // 用两个 32 位数拼成 64 位随机数
                unsigned long long a = rng();
                unsigned long long b = rng();
                ZT[i][j][c] = (a << 32) | b;
            }
        }
    }
}

// 落子（同时更新哈希值）
void placeStone(int x, int y, int color) {
    board[x][y] = color;
    currentHash ^= ZT[x][y][color];  // XOR 更新（O(1)）
}

// 撤子（XOR 再次等于恢复，与落子操作完全对称）
void removeStone(int x, int y, int color) {
    board[x][y] = EMPTY;
    currentHash ^= ZT[x][y][color];  // 再 XOR 一次 = 撤销
}

// 从头扫描棋盘，重新计算当前哈希值（仅在搜索开始前调用一次）
void computeHash() {
    currentHash = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (board[i][j] != EMPTY)
                currentHash ^= ZT[i][j][board[i][j]];
}

// ─────────────────────────────────────────────────────────────
// 第 6 节：迭代加深 Alpha-Beta 搜索
// ─────────────────────────────────────────────────────────────
/*
 * 【Alpha-Beta 剪枝是什么？】
 *   Minimax 搜索的优化版本。
 *   - ME  是极大节点（想要分值尽可能大）
 *   - OPP 是极小节点（想要分值尽可能小）
 *   - α = 当前路径上 ME 能保证的下界（初始 -INF）
 *   - β = 当前路径上 OPP 能保证的上界（初始 +INF）
 *   - 当 α ≥ β 时发生剪枝，不再搜索该子树
 *
 * 【迭代加深是什么？】
 *   从深度 2 开始，每轮加 2，直到时间到为止。
 *   这样即使时间中途耗尽，也总有一个较浅深度的完整结果作为保底。
 *
 * 【历史启发（History Heuristic）】
 *   记录哪些位置在搜索中频繁引发剪枝（说明是好棋），
 *   下次优先搜索这些位置，从而更快产生剪枝，提升搜索效率。
 */

// 历史启发分表（记录各棋盘位置的历史价值）
int histScore[SIZE][SIZE];

// 清空搜索状态（每次开始新的主搜索前调用）
void clearSearchState() {
    memset(TT,        0, sizeof(TT));
    memset(histScore, 0, sizeof(histScore));
}

/*
 * 带置换表的走法生成
 * 在普通走法列表基础上，将置换表中记录的"最优走法"提到首位，
 * 使最好的候选最先被搜索，提高剪枝效率。
 */
vector<Move> genMovesAB(int color) {
    // 查置换表，看当前棋面是否有历史最优走法
    int ttIdx = (int)(currentHash & (TT_SIZE - 1));
    int ttBestX = -1, ttBestY = -1;
    if (TT[ttIdx].hash == currentHash && TT[ttIdx].bestX >= 0) {
        ttBestX = TT[ttIdx].bestX;
        ttBestY = TT[ttIdx].bestY;
    }

    auto moves = genMoves(color, histScore, 20);

    // 将置换表最优走法提到第一位
    for (int i = 0; i < (int)moves.size(); i++) {
        if (moves[i].x == ttBestX && moves[i].y == ttBestY) {
            swap(moves[i], moves[0]);
            break;
        }
    }
    return moves;
}

/*
 * Alpha-Beta 递归搜索
 *
 * @param color   当前落子方（ME 或 OPP）
 * @param depth   剩余搜索深度（0 = 叶节点，调用静态估值）
 * @param alpha   ME 能保证的下界
 * @param beta    OPP 能保证的上界
 * @return        当前棋面的评估值（ME 视角，正值有利）
 */
int alphaBeta(int color, int depth, int alpha, int beta) {
    // ── 置换表查询 ──
    int ttIdx = (int)(currentHash & (TT_SIZE - 1));
    TTEntry &te = TT[ttIdx];
    if (te.hash == currentHash && te.depth >= depth) {
        if (te.flag == 0) return te.score;                   // 精确值，直接复用
        if (te.flag == 1 && te.score >= beta)  return te.score;  // 下界剪枝
        if (te.flag == 2 && te.score <= alpha) return te.score;  // 上界剪枝
    }

    // ── 叶节点：返回静态估值 ──
    if (depth == 0) {
        int s = evalBoard();
        TT[ttIdx] = {currentHash, s, 0, 0, -1, -1};
        return s;
    }

    // ── 生成候选走法 ──
    auto moves = genMovesAB(color);
    if (moves.empty()) {
        int s = evalBoard();
        TT[ttIdx] = {currentHash, s, depth, 0, -1, -1};
        return s;
    }

    bool isMax = (color == ME);  // ME 是最大化节点，OPP 是最小化节点
    int best   = isMax ? -INF : INF;
    int bestX  = -1, bestY = -1;

    for (int i = 0; i < (int)moves.size(); i++) {
        int mx = moves[i].x, my = moves[i].y;

        placeStone(mx, my, color);  // 落子（同时更新哈希）

        // 若落子后直接赢了，立即返回（无需继续搜索）
        if (checkWin(mx, my, color)) {
            removeStone(mx, my, color);
            // 加 depth 奖励"尽快赢"（深度越大说明离根越近 → 更快）
            int s = isMax ? (SCORE_FIVE + depth) : -(SCORE_FIVE + depth);
            TT[ttIdx] = {currentHash, s, depth, 0, mx, my};
            return s;
        }

        int s = alphaBeta(3 - color, depth - 1, alpha, beta);  // 递归搜索
        removeStone(mx, my, color);  // 撤子（同时恢复哈希）

        // ── 更新最优值和剪枝窗口 ──
        if (isMax) {
            if (s > best) { best = s; bestX = mx; bestY = my; }
            alpha = max(alpha, s);
        } else {
            if (s < best) { best = s; bestX = mx; bestY = my; }
            beta = min(beta, s);
        }

        // ── Alpha-Beta 剪枝：窗口关闭，后续走法不可能更好 ──
        if (alpha >= beta) {
            histScore[mx][my] += depth * depth;  // 更新历史启发（引发剪枝的位置加分）
            int flag = isMax ? 1 : 2;
            TT[ttIdx] = {currentHash, best, depth, flag, bestX, bestY};
            return best;
        }
    }

    TT[ttIdx] = {currentHash, best, depth, 0, bestX, bestY};
    return best;
}

/*
 * 迭代加深入口
 * 返回本次搜索选出的最佳落点 (x, y)
 */
pair<int, int> iterDeepSearch() {
    computeHash();      // 重新计算棋面哈希
    clearSearchState(); // 清空上次遗留的置换表和历史表

    auto cands = genMoves(ME, nullptr, 20);
    if (cands.empty()) {
        // 极端情况兜底（棋盘接近全满）
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                if (board[i][j] == EMPTY) return {i, j};
        return {0, 0};
    }

    pair<int, int> bestMove = {cands[0].x, cands[0].y};  // 默认取评分最高的走法

    // 从深度 2 逐步加深，时间到就停止
    for (int depth = 2; depth <= 8; depth += 2) {
        if (getElapsed() > TIME_LIMIT * 0.45) break;  // 已用超 45% 时间，停止加深

        int alpha = -INF, beta = INF;
        int bestScore = -INF;

        for (int i = 0; i < (int)cands.size(); i++) {
            if (getElapsed() > TIME_LIMIT * 0.50) break;  // 单次深度搜索超时保护

            int mx = cands[i].x, my = cands[i].y;
            placeStone(mx, my, ME);
            if (checkWin(mx, my, ME)) {
                removeStone(mx, my, ME);
                return {mx, my};  // 发现立即赢棋，直接返回
            }
            int s = alphaBeta(OPP, depth - 1, alpha, beta);
            removeStone(mx, my, ME);

            if (s > bestScore) { bestScore = s; bestMove = {mx, my}; }
            alpha = max(alpha, s);
        }
    }
    return bestMove;
}

// ─────────────────────────────────────────────────────────────
// 第 7 节：MCTS 蒙特卡洛树搜索
// ─────────────────────────────────────────────────────────────
/*
 * 【MCTS 四步骤】
 *
 *  ① Selection（选择）
 *     从根节点出发，用 UCB1 公式反复选最优子节点，
 *     直到遇到一个"还有未尝试走法"的节点（即未完全展开）。
 *
 *  ② Expansion（扩展）
 *     从未尝试走法中随机选一个，创建新子节点。
 *
 *  ③ Simulation（模拟/Rollout）
 *     从新节点出发，用启发式随机走棋，模拟到终局，
 *     得到一个胜负结果。
 *
 *  ④ Backpropagation（反传）
 *     把模拟结果沿路径传回根节点，更新每个节点的 visits 和 wins。
 *
 * 【UCB1 公式】
 *     UCB1 = wins/visits  +  C × sqrt(ln(父visits) / visits)
 *            ↑ 开发（选胜率高的）   ↑ 探索（选访问少的）
 *   C = sqrt(2) ≈ 1.414，平衡探索与开发
 *
 * 【落子/撤销方案】
 *   不存储棋面快照，而是在 Selection 时直接在真实棋盘落子，
 *   Backpropagation 结束后沿父链撤销所有落子，恢复原始棋盘。
 *   内存占用小，棋盘状态自动一致。
 */

// ── MCTS 节点结构 ──
struct MCTSNode {
    int moveX, moveY;        // 到达此节点的走法坐标（根节点为 -1）
    int playerJustMoved;     // 刚刚走棋的玩家（落子方）
    int visits;              // 该节点被访问的总次数
    int wins;                // 该节点对应玩家的胜利次数

    int parentIdx;           // 父节点在节点池中的索引（根节点为 -1）

    // 子节点列表（扩展后的子节点）
    int childCount;
    int children[25];        // 子节点索引（最多 25 个）

    // 未尝试的候选走法（待扩展的走法）
    int untriedCount;
    int untriedX[25], untriedY[25];  // 待扩展走法的坐标
};

// 节点内存池（一次性分配，避免频繁 new/delete，提高效率）
const int MCTS_POOL_SIZE = 35000;
MCTSNode nodePool[MCTS_POOL_SIZE];
int poolUsed = 0;  // 已使用节点数量

/*
 * 在节点池中创建一个新节点
 * @param parentIdx       父节点索引
 * @param moveX/moveY     到达此节点的走法
 * @param playerJustMoved 刚走棋的玩家
 * @param candidates      该节点的候选走法（作为 untried 列表）
 * @return 新节点在节点池中的索引，-1 表示池已满
 */
int createNode(int parentIdx, int moveX, int moveY, int playerJustMoved,
               const vector<Move> &candidates) {
    if (poolUsed >= MCTS_POOL_SIZE) return -1;  // 节点池已满

    MCTSNode &node = nodePool[poolUsed];
    node.moveX          = moveX;
    node.moveY          = moveY;
    node.playerJustMoved = playerJustMoved;
    node.visits         = 0;
    node.wins           = 0;
    node.parentIdx      = parentIdx;
    node.childCount     = 0;
    node.untriedCount   = 0;

    // 把候选走法装入 untried 列表
    for (int i = 0; i < (int)candidates.size() && node.untriedCount < 25; i++) {
        node.untriedX[node.untriedCount] = candidates[i].x;
        node.untriedY[node.untriedCount] = candidates[i].y;
        node.untriedCount++;
    }

    return poolUsed++;  // 返回新节点索引，并将计数器+1
}

/*
 * UCB1 选择最优子节点
 * 遍历所有子节点，选 UCB1 值最大的那个
 */
int selectBestChild(int nodeIdx) {
    MCTSNode &parent = nodePool[nodeIdx];
    const double C = 1.414;  // 探索常数，通常取 sqrt(2)
    double logParentVisits = log((double)max(1, parent.visits));

    int    bestChild = -1;
    double bestValue = -1e18;

    for (int i = 0; i < parent.childCount; i++) {
        int ci = parent.children[i];
        if (ci < 0) continue;
        MCTSNode &child = nodePool[ci];

        if (child.visits == 0) return ci;  // 未访问的子节点优先选（需要探索）

        // UCB1 = 胜率（开发项）+ 探索奖励（探索项）
        double ucb1 = (double)child.wins / child.visits
                    + C * sqrt(logParentVisits / child.visits);
        if (ucb1 > bestValue) {
            bestValue = ucb1;
            bestChild = ci;
        }
    }
    return bestChild;
}

/*
 * 快速模拟（Rollout / Simulation）
 * 从当前棋面出发，用启发式随机走棋，模拟到终局。
 * 优先级：自己能赢 > 阻止对手赢 > 按评分随机选前几个候选之一
 *
 * 注意：此函数会修改 board，调用方须在调用前备份、调用后恢复。
 *
 * @param currentPlayer 当前该谁走棋
 * @return 胜利方（ME 或 OPP），0 表示平局/未分出胜负
 */
int simulate(int currentPlayer) {
    for (int step = 0; step < 40; step++) {
        int wx, wy;

        // 优先：自己有赢棋点就走
        if (getWinMove(currentPlayer, wx, wy)) {
            board[wx][wy] = currentPlayer;
            return currentPlayer;
        }

        // 其次：堵住对手赢棋
        int opponent = 3 - currentPlayer;
        if (getWinMove(opponent, wx, wy)) {
            board[wx][wy] = currentPlayer;
            currentPlayer = opponent;
            continue;
        }

        // 最后：从评分较高的前 8 个候选中随机选一个
        auto moves = genMoves(currentPlayer, nullptr, 8);
        if (moves.empty()) break;
        int pick = rand() % min(5, (int)moves.size());
        board[moves[pick].x][moves[pick].y] = currentPlayer;
        currentPlayer = opponent;
    }
    return 0;  // 达到步数上限，视为平局
}

/*
 * MCTS 主搜索
 * 在 deadline 秒之前，不断进行 MCTS 迭代，最后选访问次数最多的子节点。
 *
 * @param deadline 截止时间（秒），超过此时间立即停止迭代
 * @return 最佳落点 (x, y)
 */
pair<int, int> mctsSearch(double deadline) {
    poolUsed = 0;  // 重置节点池（从头使用）

    // 初始化根节点：上一步是对方走的（playerJustMoved = OPP），下一步 ME 走
    auto rootCands = genMoves(ME, nullptr, 25);
    if (rootCands.empty()) return {-1, -1};
    int rootIdx = createNode(-1, -1, -1, OPP, rootCands);

    // ── MCTS 主循环 ──
    while (getElapsed() < deadline && poolUsed < MCTS_POOL_SIZE - 5) {

        // ══ 阶段 1：Selection（选择）══
        // 从根节点出发，沿 UCB1 最优路径向下，直到找到未完全展开的节点
        int nodeIdx      = rootIdx;
        int currentPlayer = ME;  // 从根出发，ME 先走

        while (nodePool[nodeIdx].untriedCount == 0 &&
               nodePool[nodeIdx].childCount > 0) {
            int next = selectBestChild(nodeIdx);
            if (next < 0) break;
            // 在真实棋盘上落子（Selection 阶段落子，Backpropagation 时撤销）
            board[nodePool[next].moveX][nodePool[next].moveY] = currentPlayer;
            currentPlayer = 3 - currentPlayer;  // 切换到下一个玩家
            nodeIdx = next;
        }

        // ══ 阶段 2：Expansion（扩展）══
        int simResult = 0;  // 模拟结果（胜利方）
        if (nodePool[nodeIdx].untriedCount > 0) {
            // 随机选一个未尝试的走法
            int idx = rand() % nodePool[nodeIdx].untriedCount;
            int nx  = nodePool[nodeIdx].untriedX[idx];
            int ny  = nodePool[nodeIdx].untriedY[idx];

            // 将该走法从 untried 列表中移除（交换到末尾再缩减）
            int last = nodePool[nodeIdx].untriedCount - 1;
            nodePool[nodeIdx].untriedX[idx] = nodePool[nodeIdx].untriedX[last];
            nodePool[nodeIdx].untriedY[idx] = nodePool[nodeIdx].untriedY[last];
            nodePool[nodeIdx].untriedCount--;

            board[nx][ny] = currentPlayer;  // 落子，扩展新节点

            if (checkWin(nx, ny, currentPlayer)) {
                // 扩展的走法直接赢了 → 创建终止子节点
                vector<Move> noMore;  // 已赢，不需要候选走法
                int childIdx = createNode(nodeIdx, nx, ny, currentPlayer, noMore);
                if (childIdx >= 0) {
                    nodePool[nodeIdx].children[nodePool[nodeIdx].childCount++] = childIdx;
                    nodeIdx = childIdx;
                }
                simResult = currentPlayer;  // 记录胜利方

            } else {
                // 未立即赢，扩展子节点并模拟
                int nextPlayer = 3 - currentPlayer;
                auto childCands = genMoves(nextPlayer, nullptr, 25);
                int childIdx = createNode(nodeIdx, nx, ny, currentPlayer, childCands);
                if (childIdx >= 0) {
                    nodePool[nodeIdx].children[nodePool[nodeIdx].childCount++] = childIdx;
                    nodeIdx = childIdx;

                    // ══ 阶段 3：Simulation（模拟）══
                    // 备份棋盘 → 随机模拟 → 恢复棋盘
                    int boardBackup[SIZE][SIZE];
                    memcpy(boardBackup, board, sizeof(board));
                    simResult = simulate(nextPlayer);
                    memcpy(board, boardBackup, sizeof(board));
                }
            }
        }

        // ══ 阶段 4：Backpropagation（反传）+ 撤销落子 ══
        // 从当前节点沿父链向上传播结果，同时撤销 Selection 阶段落下的棋子
        int cur = nodeIdx;
        while (cur >= 0) {
            nodePool[cur].visits++;
            // wins 记录"到达此节点的玩家"的胜利次数
            if (nodePool[cur].playerJustMoved == simResult)
                nodePool[cur].wins++;

            if (cur == rootIdx) break;  // 到达根节点，停止

            // 撤销这个节点的落子，恢复棋盘状态
            board[nodePool[cur].moveX][nodePool[cur].moveY] = EMPTY;
            cur = nodePool[cur].parentIdx;
        }
        // 此时棋盘已完全恢复到 MCTS 开始前的状态
    }

    // ── 选出最优走法：访问次数最多的子节点 ──
    // （访问次数最多说明该走法在模拟中表现最稳定）
    MCTSNode &rootNode = nodePool[rootIdx];
    int bestChildIdx = -1;
    int maxVisits    = 0;
    for (int i = 0; i < rootNode.childCount; i++) {
        int ci = rootNode.children[i];
        if (ci < 0) continue;
        if (nodePool[ci].visits > maxVisits) {
            maxVisits    = nodePool[ci].visits;
            bestChildIdx = ci;
        }
    }

    if (bestChildIdx >= 0)
        return {nodePool[bestChildIdx].moveX, nodePool[bestChildIdx].moveY};
    return {-1, -1};
}

// ─────────────────────────────────────────────────────────────
// 第 8 节：主函数
// ─────────────────────────────────────────────────────────────

/*
 * 找一个保底合法走法（防止极端情况下无法输出）
 * 优先选有邻近棋子的空格，实在没有就随便一个空格
 */
pair<int, int> fallbackMove() {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (board[i][j] == EMPTY && hasNeighbor(i, j))
                return {i, j};
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (board[i][j] == EMPTY)
                return {i, j};
    return {0, 0};
}

int main() {
    // 初始化
    g_startTime = clock();
    srand((unsigned)time(nullptr));  // 随机数种子（每次运行不同）
    memset(board, 0, sizeof(board));
    initZobrist();

    int n, x, y;
    cin >> n;

    // ── 读取历史棋局（与 Gomoku_simple.cpp 格式完全一致）──
    // board[x][y] = OPP → 对方棋子
    // board[x][y] = ME  → 我方棋子
    for (int i = 0; i < n - 1; i++) {
        cin >> x >> y; if (x != -1) board[x][y] = OPP;  // 历史回合：对方落子
        cin >> x >> y; if (x != -1) board[x][y] = ME;   // 历史回合：我方落子
    }
    cin >> x >> y;  // 本回合：对方最新落子
    if (x != -1) board[x][y] = OPP;

    int new_x = -1, new_y = -1;

    // ── 第 1 回合：Swap1 规则特殊处理 ──
    if (n == 1) {
        if (x == -1) {
            // 我方先手：走天元（棋盘正中心），常规开局
            new_x = SIZE / 2;
            new_y = SIZE / 2;
        } else {
            // 我方后手：判断是否需要换手
            // 如果对方开局位置在中心附近（曼哈顿距离≤2），说明是强开局，换手占优
            int dist = abs(x - SIZE / 2) + abs(y - SIZE / 2);
            if (dist <= 2) {
                new_x = -1; new_y = -1;  // 输出 (-1,-1) 表示换手
            } else {
                new_x = SIZE / 2; new_y = SIZE / 2;  // 不换手，走天元
            }
        }
        printf("%d %d\n", new_x, new_y);
        return 0;
    }

    // ────────────────────────────────────────────────────────
    // n ≥ 2：5 层决策漏斗
    // 每层找到答案就跳转到 OUTPUT 输出，不进入下一层
    // ────────────────────────────────────────────────────────

    // ── 层 1：即时胜/防（最高优先级）──
    {
        int wx, wy;
        if (getWinMove(ME, wx, wy))  { new_x = wx; new_y = wy; goto OUTPUT; }  // 我能赢，立即赢
        if (getWinMove(OPP, wx, wy)) { new_x = wx; new_y = wy; goto OUTPUT; }  // 对手能赢，立即堵
    }

    // ── 层 2：双威胁检测（活四 / 双冲四 = 必胜局面）──
    {
        auto cands = genMoves(ME, nullptr, 20);
        for (int i = 0; i < (int)cands.size(); i++) {
            int s = evalPoint(cands[i].x, cands[i].y, ME);
            // 活四（SCORE_L4）或双冲四（≥2×SCORE_R4）→ 下一步必赢
            if (s >= SCORE_L4 || s >= 2 * SCORE_R4) {
                new_x = cands[i].x;
                new_y = cands[i].y;
                goto OUTPUT;
            }
        }
    }

    // ── 层 3：VCF 必杀搜索 ──
    {
        int wx = -1, wy = -1;
        if (vcfCheck(wx, wy) != 0) { new_x = wx; new_y = wy; goto OUTPUT; }
    }

    // ── 层 4：迭代加深 Alpha-Beta ──
    {
        pair<int, int> abResult = iterDeepSearch();
        new_x = abResult.first;   // 先用 AB 结果保底
        new_y = abResult.second;
    }

    // ── 层 5：MCTS 精化搜索（利用剩余时间进一步优化）──
    if (getElapsed() < TIME_LIMIT * 0.80) {
        pair<int, int> mctsResult = mctsSearch(TIME_LIMIT * 0.92);
        int mx = mctsResult.first, my = mctsResult.second;
        // 只有当 MCTS 返回合法位置时才采用（安全检查）
        if (mx >= 0 && mx < SIZE && my >= 0 && my < SIZE && board[mx][my] == EMPTY) {
            new_x = mx;
            new_y = my;
        }
    }

OUTPUT:
    // ── 安全兜底：确保输出的坐标合法 ──
    // 若上面所有层都出了问题（理论上不应该发生），则用 fallback
    if (new_x < 0 || new_x >= SIZE || new_y < 0 || new_y >= SIZE
        || board[new_x][new_y] != EMPTY) {
        pair<int, int> fb = fallbackMove();
        new_x = fb.first;
        new_y = fb.second;
    }

    printf("%d %d\n", new_x, new_y);
    return 0;
}
