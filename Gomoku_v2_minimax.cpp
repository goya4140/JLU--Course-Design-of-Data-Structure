/*
 * ============================================================
 *   五子棋 AI - 第二阶段：Minimax 博弈树搜索
 *   适用平台：Botzone (Gomoku-Swap1)
 *   编译命令：g++ -O2 -std=c++17 -o gomoku Gomoku_v2_minimax.cpp
 * ============================================================
 *
 *   【算法简介】
 *   贪心只看"当前一步"哪里分高就下哪里，无法预判对手的反击。
 *   Minimax 通过递归展开博弈树，向前看 SEARCH_DEPTH 步（本版本=4），
 *   假设对手每步都下最优（Min层），我方每步也选最优（Max层），
 *   从而找出真正"经得起对手最强反击"的最佳走法。
 *
 *   【核心函数说明】
 *   evaluateLine()    - 评估一条线上的棋形得分
 *   evaluateBoard()   - 评估整个局面（我方分 - 对方分）
 *   checkWin()        - 判断某步落子是否形成五连
 *   generateMoves()   - 生成候选走法（带邻近剪枝 + 排序）
 *   minimax()         - Minimax 递归搜索核心
 *   findBestMove()    - 顶层入口，返回最佳走法坐标
 *
 *   【与第一阶段的区别】
 *   - 第一阶段：对每个空位打分，选分最高的（贪心）
 *   - 第二阶段：对每个空位"假设我下了"，再看对手怎么应，
 *              再看我方怎么应对手……递归 4 层后用估值函数评分
 */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
using namespace std;

// ===================== 常量定义 =====================

const int SIZE = 15;               // 棋盘大小
const int SEARCH_DEPTH = 4;        // 搜索深度（向前看4步 = 双方各走2回合）
const int INF = 10000000;          // 无穷大
const int WIN_SCORE = 1000000;     // 五连的分值

// 棋盘：0=空, 1=我方, 2=对方
// 注意：内部用 1 和 2 表示双方，比 -1/1 更方便做数组索引
int board[SIZE][SIZE];

// 计时器，防止超时
clock_t start_time;
const double TIME_LIMIT = 0.90;    // 秒，留0.1秒安全余量
bool timeout_flag = false;         // 超时标记

// 4个方向向量：横、竖、主对角线、副对角线
const int DX[4] = {0, 1, 1, 1};
const int DY[4] = {1, 0, 1, -1};

// ===================== 工具函数 =====================

/* 判断坐标是否在棋盘内 */
inline bool inBoard(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

/* 检查是否超时 */
inline bool isTimeout() {
    return (double)(clock() - start_time) / CLOCKS_PER_SEC > TIME_LIMIT;
}

// ===================== 胜负判定 =====================

/*
 * checkWin(x, y, role)
 * 判断在 (x,y) 处的 role 棋子是否形成了五连
 * role: 1=我方, 2=对方
 * 返回: true=形成五连, false=未形成
 *
 * 思路：沿4个方向，从(x,y)向两边延伸，数同色连子数
 */
bool checkWin(int x, int y, int role) {
    for (int d = 0; d < 4; d++) {
        int count = 1;  // 包含(x,y)自身
        // 正方向
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == role) {
            count++;
            nx += DX[d]; ny += DY[d];
        }
        // 反方向
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == role) {
            count++;
            nx -= DX[d]; ny -= DY[d];
        }
        if (count >= 5) return true;
    }
    return false;
}

// ===================== 估值函数 =====================

/*
 * evaluateLine(board, x, y, dx, dy, role)
 *
 * 从(x,y)沿方向(dx,dy)扫描一条线，识别属于 role 的棋形并打分。
 * 这是估值系统的核心——通过"连子数 + 空端数"组合来判定棋形。
 *
 *   连5 = 直接获胜              → 1000000
 *   活4 = 两端都空的4连（必胜）  → 100000
 *   冲4 = 一端被堵的4连          → 8000
 *   活3 = 两端都空的3连          → 1000
 *   眠3 = 一端被堵的3连          → 100
 *   活2 = 两端都空的2连          → 100
 *   眠2 = 一端被堵的2连          → 10
 */
int scorePattern(int count, int openEnds) {
    // count=连子数, openEnds=空端数(0/1/2)
    if (count >= 5) return WIN_SCORE;

    if (count == 4) {
        if (openEnds == 2) return 100000;   // 活四
        if (openEnds == 1) return 8000;     // 冲四
        return 0;                            // 死四（两端都堵）
    }
    if (count == 3) {
        if (openEnds == 2) return 1000;     // 活三
        if (openEnds == 1) return 100;      // 眠三
        return 0;
    }
    if (count == 2) {
        if (openEnds == 2) return 100;      // 活二
        if (openEnds == 1) return 10;       // 眠二
        return 0;
    }
    if (count == 1) {
        if (openEnds == 2) return 1;        // 活一
        return 0;
    }
    return 0;
}

/*
 * evaluateBoard()
 *
 * 评估当前整个棋盘局面的得分。
 * 返回值 > 0 表示对我方有利，< 0 表示对对方有利。
 *
 * 实现方式：
 *   遍历棋盘上每个位置，沿4个方向扫描，
 *   统计我方(role=1)和对方(role=2)分别能形成的棋形，
 *   然后返回 我方总分 - 对方总分。
 *
 * 为避免重复计算，只从每条线的"起点"开始扫：
 *   横线：从(i,0)开始
 *   竖线：从(0,j)开始
 *   主对角线：从第0行或第0列开始
 *   副对角线：从第0行或最后一列开始
 */
int evaluateForRole(int role) {
    int score = 0;
    int opponent = (role == 1) ? 2 : 1;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != role) continue;

            // 沿4个方向检查以 (i,j) 为起点的棋形
            for (int d = 0; d < 4; d++) {
                // 只在 "前一格不是同色" 时才从这里开始计数
                // 避免同一条线被重复计算
                int px = i - DX[d], py = j - DY[d];
                if (inBoard(px, py) && board[px][py] == role)
                    continue;  // 前面还有同色子，说明不是这条线的起点

                // 从 (i,j) 开始向正方向数连子
                int count = 0;
                int nx = i, ny = j;
                while (inBoard(nx, ny) && board[nx][ny] == role) {
                    count++;
                    nx += DX[d]; ny += DY[d];
                }

                // 检查两端是否为空
                int openEnds = 0;
                // 正方向尽头
                if (inBoard(nx, ny) && board[nx][ny] == 0)
                    openEnds++;
                // 反方向尽头
                if (inBoard(px, py) && board[px][py] == 0)
                    openEnds++;

                score += scorePattern(count, openEnds);
            }
        }
    }
    return score;
}

int evaluateBoard() {
    return evaluateForRole(1) - evaluateForRole(2);
    // 正值 = 我方优势，负值 = 对方优势
}

// ===================== 候选走法生成 =====================

/*
 * 候选走法结构体
 * 存储一个候选位置的坐标和"预估分"（用于排序）
 */
struct Move {
    int x, y;
    int priority;  // 预估得分，用于走法排序（好的走法先搜）
};

/*
 * hasNeighbor(x, y, dist)
 * 判断(x,y)周围 dist 格内是否有棋子
 * 这是候选走法剪枝的关键：只考虑已有棋子附近的空位
 * 大幅减少搜索分支（从 ~100 减到 ~20-30）
 */
bool hasNeighbor(int x, int y, int dist) {
    for (int i = -dist; i <= dist; i++)
        for (int j = -dist; j <= dist; j++) {
            int nx = x + i, ny = y + j;
            if (inBoard(nx, ny) && board[nx][ny] != 0)
                return true;
        }
    return false;
}

/*
 * quickEval(x, y)
 * 快速估算在(x,y)落子的价值，用于候选走法排序
 * 等价于第一阶段的贪心评估（进攻分 + 防守分）
 */
int quickEval(int x, int y) {
    int myScore = 0, oppScore = 0;
    for (int d = 0; d < 4; d++) {
        // 我方落子的价值
        int count = 1, openEnds = 0;
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 1) { count++; nx += DX[d]; ny += DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 1) { count++; nx -= DX[d]; ny -= DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        myScore += scorePattern(count, openEnds);

        // 对方落子的价值
        count = 1; openEnds = 0;
        nx = x + DX[d]; ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 2) { count++; nx += DX[d]; ny += DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 2) { count++; nx -= DX[d]; ny -= DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        oppScore += scorePattern(count, openEnds);
    }
    return myScore + oppScore;  // 进攻和防守都高的点优先搜
}

/*
 * generateMoves()
 * 生成当前局面下的候选走法列表
 * 
 * 策略：
 * 1. 只考虑已有棋子周围 2 格内的空位（邻近剪枝）
 * 2. 按 quickEval 分数从高到低排序（走法排序，为 Alpha-Beta 做准备）
 * 3. 最多取前 MAX_MOVES 个候选（控制分支因子）
 */
vector<Move> generateMoves() {
    vector<Move> moves;
    const int MAX_MOVES = 20;  // 每层最多搜20个候选

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != 0) continue;
            if (!hasNeighbor(i, j, 2)) continue;

            Move m;
            m.x = i;
            m.y = j;
            m.priority = quickEval(i, j);
            moves.push_back(m);
        }
    }

    // 按预估分从高到低排序
    sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
        return a.priority > b.priority;
    });

    // 只保留前 MAX_MOVES 个
    if ((int)moves.size() > MAX_MOVES) {
        moves.resize(MAX_MOVES);
    }

    return moves;
}

// ===================== Minimax 搜索核心 =====================

/*
 * minimax(depth, isMaxPlayer, lastX, lastY)
 *
 * 参数说明：
 *   depth        - 剩余搜索深度（每层减1，到0用估值函数）
 *   isMaxPlayer  - true=当前是我方走（取max），false=当前是对方走（取min）
 *   lastX, lastY - 上一步落子的坐标（用于快速判断上一步是否获胜）
 *
 * 返回值：
 *   当前局面在最优策略下的评估分
 *
 * 【递归逻辑】
 *   1. 如果上一步形成了五连 → 直接返回胜/负分
 *   2. 如果深度为 0 → 返回估值函数
 *   3. 生成候选走法
 *   4. 对每个走法：落子 → 递归 → 撤销
 *   5. Max层取最大值，Min层取最小值
 */
int minimax(int depth, bool isMaxPlayer, int lastX, int lastY) {
    // 超时保护：立即返回
    if (timeout_flag) return 0;
    if (isTimeout()) {
        timeout_flag = true;
        return 0;
    }

    // 终止条件 1：上一步对方/我方五连获胜
    if (lastX >= 0) {
        int lastRole = board[lastX][lastY];
        if (checkWin(lastX, lastY, lastRole)) {
            // 上一步落子的人赢了
            // 如果当前是 Max 层（轮我方走），说明刚才是对方走的，对方赢了 → 返回负分
            // 如果当前是 Min 层（轮对方走），说明刚才是我方走的，我方赢了 → 返回正分
            if (isMaxPlayer) {
                // 上一步是对方下的（Min层的上一层），对方赢了
                return -(WIN_SCORE + depth);  // 加 depth 使得"更快赢"的路径分更高
            } else {
                // 上一步是我方下的（Max层的上一层），我方赢了
                return WIN_SCORE + depth;
            }
        }
    }

    // 终止条件 2：搜索深度耗尽
    if (depth == 0) {
        return evaluateBoard();
    }

    // 生成候选走法
    vector<Move> moves = generateMoves();

    // 没有合法走法（理论上不会出现，除非棋盘满了）
    if (moves.empty()) {
        return evaluateBoard();
    }

    if (isMaxPlayer) {
        // ===== 我方走（Max 层）=====
        // 我方会选择使分数最大的走法
        int maxEval = -INF;
        for (auto& m : moves) {
            board[m.x][m.y] = 1;                              // 我方落子
            int eval = minimax(depth - 1, false, m.x, m.y);   // 递归：下一层是对方
            board[m.x][m.y] = 0;                              // 撤销落子（回溯）

            maxEval = max(maxEval, eval);

            if (timeout_flag) break;
        }
        return maxEval;
    } else {
        // ===== 对方走（Min 层）=====
        // 对方会选择使分数最小的走法
        int minEval = INF;
        for (auto& m : moves) {
            board[m.x][m.y] = 2;                              // 对方落子
            int eval = minimax(depth - 1, true, m.x, m.y);    // 递归：下一层是我方
            board[m.x][m.y] = 0;                              // 撤销落子（回溯）

            minEval = min(minEval, eval);

            if (timeout_flag) break;
        }
        return minEval;
    }
}

// ===================== 顶层决策入口 =====================

/*
 * findBestMove(bestX, bestY)
 *
 * 在所有候选走法中，用 Minimax 搜索找出最佳的一步。
 * 结果通过 bestX, bestY 返回。
 *
 * 【额外优化：即时必胜/必守检测】
 *   在启动深度搜索之前，先做一遍快速扫描：
 *   1. 如果我方某个位置能直接五连 → 立刻下（进攻必胜）
 *   2. 如果对方某个位置能直接五连 → 立刻堵（防守必守）
 *   这两种情况不需要搜索，节省大量时间。
 */
void findBestMove(int& bestX, int& bestY) {
    vector<Move> moves = generateMoves();

    if (moves.empty()) {
        // 兜底：下天元
        bestX = 7; bestY = 7;
        return;
    }

    // ===== 快速检测：我方能否直接获胜 =====
    for (auto& m : moves) {
        board[m.x][m.y] = 1;
        if (checkWin(m.x, m.y, 1)) {
            board[m.x][m.y] = 0;
            bestX = m.x; bestY = m.y;
            return;  // 直接赢！
        }
        board[m.x][m.y] = 0;
    }

    // ===== 快速检测：对方是否有致命威胁需要堵 =====
    for (auto& m : moves) {
        board[m.x][m.y] = 2;
        if (checkWin(m.x, m.y, 2)) {
            board[m.x][m.y] = 0;
            bestX = m.x; bestY = m.y;
            return;  // 必须堵这里！
        }
        board[m.x][m.y] = 0;
    }

    // ===== Minimax 搜索 =====
    int bestScore = -INF;
    bestX = moves[0].x;
    bestY = moves[0].y;

    for (auto& m : moves) {
        board[m.x][m.y] = 1;     // 我方落子
        int score = minimax(SEARCH_DEPTH - 1, false, m.x, m.y);
        board[m.x][m.y] = 0;     // 撤销

        if (score > bestScore) {
            bestScore = score;
            bestX = m.x;
            bestY = m.y;
        }

        // 超时保护：使用目前最佳结果
        if (timeout_flag) break;

        // 发现必胜，不用再搜
        if (bestScore >= WIN_SCORE) break;
    }
}

// ===================== 主函数 =====================

int main() {
    start_time = clock();

    // 初始化棋盘
    memset(board, 0, sizeof(board));

    int n, x, y;
    cin >> n;

    // 重建历史棋盘
    // Botzone 输入格式：对方用 -1 标记，我方用 1 标记
    // 我们内部用 1=我方, 2=对方
    for (int i = 0; i < n - 1; i++) {
        cin >> x >> y;
        if (x != -1) board[x][y] = 2;   // 对方历史落子
        cin >> x >> y;
        if (x != -1) board[x][y] = 1;   // 我方历史落子
    }
    cin >> x >> y;
    if (x != -1) board[x][y] = 2;       // 对方本回合落子

    int new_x = -1, new_y = -1;

    // ===== 特殊处理：开局和换手 =====

    // 情况1：我方先手第一步（对方输入 -1 -1），棋盘上无子
    if (n == 1 && x == -1) {
        new_x = 7; new_y = 7;  // 下天元（最强开局）
    }
    // 情况2：我方后手第一回合，判断是否换手
    else if (n == 1 && x != -1) {
        // Swap1 策略：对方下在中心附近（太强势）→ 换手抢先手优势
        int dist = abs(x - 7) + abs(y - 7);
        if (dist <= 2) {
            new_x = -1; new_y = -1;  // 换手
        } else {
            // 对方下在偏远处，不换手，自己占天元
            new_x = 7; new_y = 7;
        }
    }
    // 情况3：正常对弈 → Minimax 搜索
    else {
        // 特殊情况：棋盘上完全没有棋子（例如我方先手被换手后的第一步）
        bool hasStone = false;
        for (int i = 0; i < SIZE && !hasStone; i++)
            for (int j = 0; j < SIZE && !hasStone; j++)
                if (board[i][j] != 0) hasStone = true;

        if (!hasStone) {
            new_x = 7; new_y = 7;
        } else {
            findBestMove(new_x, new_y);
        }
    }

    printf("%d %d\n", new_x, new_y);
    return 0;
}
