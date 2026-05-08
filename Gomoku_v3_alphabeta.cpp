/*
 * ============================================================
 *   五子棋 AI - 第三阶段：Alpha-Beta 剪枝
 *   适用平台：Botzone (Gomoku-Swap1)
 *   编译命令：g++ -O2 -std=c++17 -o gomoku Gomoku_v3_alphabeta.cpp
 * ============================================================
 *
 *   【从第二阶段到第三阶段的升级】
 *
 *   第二阶段 Minimax 搜 4 层需要遍历所有节点（约 16 万个）。
 *   Alpha-Beta 通过"剪枝"跳过不可能影响最终决策的分支，
 *   在走法排序良好的情况下，能把节点数从 b^d 降到约 b^(d/2)。
 *   这意味着：同样 1 秒的时间，搜索深度可从 4 提升到 6~8。
 *
 *   【本阶段新增的三个关键技术】
 *
 *   1. Alpha-Beta 剪枝
 *      维护 α（我方下界）和 β（对方上界），α≥β 时立刻剪枝。
 *
 *   2. 迭代加深（Iterative Deepening）
 *      不是一次性搜到最大深度，而是依次搜 2、4、6、8 层。
 *      好处：① 浅层结果可以改善深层的走法排序（搜得更快）
 *            ② 天然适配卡时——时间不够就用上一轮的结果
 *
 *   3. 必杀走法优先（Killer Move）
 *      如果某个走法能直接形成五连或活四，提到候选列表最前面。
 *      这类走法最容易触发剪枝，大幅减少搜索量。
 */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
using namespace std;

// ===================== 常量定义 =====================

const int SIZE = 15;
const int INF = 10000000;
const int WIN_SCORE = 1000000;
const int MAX_DEPTH = 8;           // 迭代加深的最大深度上限

int board[SIZE][SIZE];             // 0=空, 1=我方, 2=对方

clock_t start_time;
const double TIME_LIMIT = 0.90;    // 秒
bool timeout_flag = false;

const int DX[4] = {0, 1, 1, 1};
const int DY[4] = {1, 0, 1, -1};

// ===================== 工具函数 =====================

inline bool inBoard(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

inline bool isTimeout() {
    return (double)(clock() - start_time) / CLOCKS_PER_SEC > TIME_LIMIT;
}

// ===================== 胜负判定 =====================

bool checkWin(int x, int y, int role) {
    for (int d = 0; d < 4; d++) {
        int count = 1;
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == role) {
            count++; nx += DX[d]; ny += DY[d];
        }
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == role) {
            count++; nx -= DX[d]; ny -= DY[d];
        }
        if (count >= 5) return true;
    }
    return false;
}

// ===================== 估值函数 =====================

/*
 * scorePattern(count, openEnds)
 *
 * 根据连子数和空端数返回该棋形的分值。
 * 分值体系的设计原则：
 *   - 相邻等级之间至少差一个数量级，确保高级棋形绝对优先
 *   - 活棋形远高于眠棋形（两端空 vs 一端堵）
 *   - 五连分值极高，保证搜索到必胜路径时立刻选择
 *
 *   连5: 1000000    活4: 100000    冲4: 8000
 *   活3: 1000       眠3: 100
 *   活2: 100        眠2: 10
 */
int scorePattern(int count, int openEnds) {
    if (count >= 5) return WIN_SCORE;
    if (count == 4) {
        if (openEnds == 2) return 100000;
        if (openEnds == 1) return 8000;
        return 0;
    }
    if (count == 3) {
        if (openEnds == 2) return 1000;
        if (openEnds == 1) return 100;
        return 0;
    }
    if (count == 2) {
        if (openEnds == 2) return 100;
        if (openEnds == 1) return 10;
        return 0;
    }
    if (count == 1 && openEnds == 2) return 1;
    return 0;
}

/*
 * evaluateForRole(role)
 *
 * 扫描整个棋盘，统计 role 所拥有的所有棋形得分之和。
 * 关键技巧：只从每条同色连子的"起点"开始计数，避免重复。
 * 判定起点的方法：(i,j) 的前一格不是同色棋子。
 */
int evaluateForRole(int role) {
    int score = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != role) continue;
            for (int d = 0; d < 4; d++) {
                // 跳过非起点
                int px = i - DX[d], py = j - DY[d];
                if (inBoard(px, py) && board[px][py] == role) continue;

                // 从起点向正方向数连子
                int count = 0;
                int nx = i, ny = j;
                while (inBoard(nx, ny) && board[nx][ny] == role) {
                    count++; nx += DX[d]; ny += DY[d];
                }

                // 计算空端数
                int openEnds = 0;
                if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
                if (inBoard(px, py) && board[px][py] == 0) openEnds++;

                score += scorePattern(count, openEnds);
            }
        }
    }
    return score;
}

/* 全局面估值：正值=我方优势，负值=对方优势 */
int evaluateBoard() {
    return evaluateForRole(1) - evaluateForRole(2);
}

// ===================== 候选走法 =====================

struct Move {
    int x, y;
    int priority;
};

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
 * 快速估算在(x,y)落子的价值，用于候选走法排序。
 *
 * 【第三阶段改进】增加了"必杀检测"：
 *   如果我方在此处落子能形成五连 → 优先级设为极高
 *   如果对方在此处落子能形成五连 → 优先级也设为极高（必须堵）
 *   这些走法排在最前面，能极大加速 Alpha-Beta 剪枝。
 */
int quickEval(int x, int y) {
    int myScore = 0, oppScore = 0;

    for (int d = 0; d < 4; d++) {
        // 评估我方棋形
        int count = 1, openEnds = 0;
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 1) { count++; nx += DX[d]; ny += DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 1) { count++; nx -= DX[d]; ny -= DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        myScore += scorePattern(count, openEnds);

        // 评估对方棋形
        count = 1; openEnds = 0;
        nx = x + DX[d]; ny = y + DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 2) { count++; nx += DX[d]; ny += DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && board[nx][ny] == 2) { count++; nx -= DX[d]; ny -= DY[d]; }
        if (inBoard(nx, ny) && board[nx][ny] == 0) openEnds++;
        oppScore += scorePattern(count, openEnds);
    }

    // 必杀/必守得极高优先级
    if (myScore >= WIN_SCORE) return 20000000;
    if (oppScore >= WIN_SCORE) return 10000000;

    return myScore * 2 + oppScore;  // 进攻略优先
}

/*
 * generateMoves(maxMoves)
 *
 * 生成候选走法，按优先级排序，最多取 maxMoves 个。
 * maxMoves 的设置：
 *   - 根节点和浅层：取多一些（25个），保证不漏好棋
 *   - 深层：取少一些（15个），加速搜索
 */
vector<Move> generateMoves(int maxMoves = 20) {
    vector<Move> moves;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] != 0) continue;
            if (!hasNeighbor(i, j, 2)) continue;

            Move m;
            m.x = i; m.y = j;
            m.priority = quickEval(i, j);
            moves.push_back(m);
        }
    }

    sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
        return a.priority > b.priority;
    });

    if ((int)moves.size() > maxMoves)
        moves.resize(maxMoves);

    return moves;
}

// ===================== Alpha-Beta 搜索核心 =====================

/*
 * alphaBeta(depth, alpha, beta, isMaxPlayer, lastX, lastY)
 *
 *   和 Minimax 的唯一区别：多了 alpha 和 beta 两个参数。
 *
 *   alpha = 我方在祖先节点中已找到的最优下界（"我至少能拿这么多"）
 *   beta  = 对方在祖先节点中已找到的最优上界（"对方至多让我拿这么多"）
 *
 *   剪枝条件：alpha >= beta
 *     → 对于 Max 层：如果当前最佳值 >= beta，对方不会让我走到这里
 *     → 对于 Min 层：如果当前最佳值 <= alpha，我方不会走到这里
 *
 *   这就像"双向夹逼"：alpha 从下面往上推，beta 从上面往下压，
 *   一旦交叉就说明这条路不可能是最优解，直接跳过。
 *
 *   【和 Minimax 的代码对比】
 *   Minimax:                          Alpha-Beta:
 *   maxEval = max(maxEval, eval)      maxEval = max(maxEval, eval)
 *                                     alpha = max(alpha, eval)     ← 新增
 *                                     if (alpha >= beta) break     ← 新增（剪枝！）
 */
int alphaBeta(int depth, int alpha, int beta, bool isMaxPlayer, int lastX, int lastY) {
    // 超时保护
    if (timeout_flag) return 0;
    if (isTimeout()) { timeout_flag = true; return 0; }

    // 终止条件 1：上一步形成五连
    if (lastX >= 0) {
        int lastRole = board[lastX][lastY];
        if (checkWin(lastX, lastY, lastRole)) {
            if (isMaxPlayer) {
                return -(WIN_SCORE + depth);  // 对方赢了
            } else {
                return WIN_SCORE + depth;     // 我方赢了
            }
        }
    }

    // 终止条件 2：深度耗尽
    if (depth == 0) {
        return evaluateBoard();
    }

    // 深层用更少的候选走法加速搜索
    int maxMoves = (depth >= 4) ? 25 : 15;
    vector<Move> moves = generateMoves(maxMoves);

    if (moves.empty()) {
        return evaluateBoard();
    }

    if (isMaxPlayer) {
        // ===== 我方走（Max 层）=====
        int maxEval = -INF;
        for (auto& m : moves) {
            board[m.x][m.y] = 1;
            int eval = alphaBeta(depth - 1, alpha, beta, false, m.x, m.y);
            board[m.x][m.y] = 0;

            maxEval = max(maxEval, eval);

            // 【Alpha-Beta 核心：更新 alpha 并检查剪枝】
            alpha = max(alpha, eval);
            if (alpha >= beta) {
                // Beta 剪枝：对方不会允许我方走到这么好的局面
                // 当前节点剩余的兄弟分支全部跳过
                break;
            }

            if (timeout_flag) break;
        }
        return maxEval;
    } else {
        // ===== 对方走（Min 层）=====
        int minEval = INF;
        for (auto& m : moves) {
            board[m.x][m.y] = 2;
            int eval = alphaBeta(depth - 1, alpha, beta, true, m.x, m.y);
            board[m.x][m.y] = 0;

            minEval = min(minEval, eval);

            // 【Alpha-Beta 核心：更新 beta 并检查剪枝】
            beta = min(beta, eval);
            if (alpha >= beta) {
                // Alpha 剪枝：我方不会选择走到这么差的局面
                break;
            }

            if (timeout_flag) break;
        }
        return minEval;
    }
}

// ===================== 迭代加深 =====================

/*
 * findBestMove(bestX, bestY)
 *
 *   【迭代加深策略（Iterative Deepening）】
 *
 *   不是直接搜深度 8，而是依次搜 2、4、6、8 层。
 *
 *   为什么？三个好处：
 *   1. 天然卡时：每轮搜完检查剩余时间，不够就用上一轮结果
 *   2. 走法排序改进：浅层找到的最佳走法，在深层优先搜索（大幅增加剪枝）
 *   3. 搜索开销很小：所有浅层搜索的总开销远小于最深一层
 *      （因为 b^2 + b^4 + b^6 << b^8，被最后一项主导）
 *
 *   实际效果：迭代加深 + Alpha-Beta + 走法排序三者结合，
 *   比单纯 Minimax 快 100 倍以上，搜索深度通常能达到 6~8。
 */
void findBestMove(int& bestX, int& bestY) {
    vector<Move> moves = generateMoves(25);

    if (moves.empty()) {
        bestX = 7; bestY = 7;
        return;
    }

    // ===== 快速检测：直接赢或必须堵 =====
    for (auto& m : moves) {
        board[m.x][m.y] = 1;
        if (checkWin(m.x, m.y, 1)) {
            board[m.x][m.y] = 0;
            bestX = m.x; bestY = m.y;
            return;
        }
        board[m.x][m.y] = 0;
    }
    for (auto& m : moves) {
        board[m.x][m.y] = 2;
        if (checkWin(m.x, m.y, 2)) {
            board[m.x][m.y] = 0;
            bestX = m.x; bestY = m.y;
            return;
        }
        board[m.x][m.y] = 0;
    }

    // ===== 迭代加深搜索 =====
    bestX = moves[0].x;
    bestY = moves[0].y;
    int bestMoveIdx = 0;      // 记录最佳走法在 moves 中的索引

    for (int depth = 2; depth <= MAX_DEPTH; depth += 2) {
        timeout_flag = false;
        int currentBestScore = -INF;
        int currentBestIdx = 0;

        /*
         * 【走法排序优化】
         * 把上一轮迭代找到的最佳走法移到列表最前面。
         * 这使得 Alpha-Beta 在新一轮搜索中更早找到好的下界，
         * 触发更多剪枝，搜索速度大幅提升。
         */
        if (bestMoveIdx > 0 && bestMoveIdx < (int)moves.size()) {
            swap(moves[0], moves[bestMoveIdx]);
        }

        for (int i = 0; i < (int)moves.size(); i++) {
            Move& m = moves[i];
            board[m.x][m.y] = 1;
            int score = alphaBeta(depth - 1, -INF, INF, false, m.x, m.y);
            board[m.x][m.y] = 0;

            if (timeout_flag) break;

            if (score > currentBestScore) {
                currentBestScore = score;
                currentBestIdx = i;
            }

            // 必胜，不用再搜
            if (currentBestScore >= WIN_SCORE) break;
        }

        // 本轮搜索完整完成（未超时），更新最终结果
        if (!timeout_flag) {
            bestX = moves[currentBestIdx].x;
            bestY = moves[currentBestIdx].y;
            bestMoveIdx = currentBestIdx;

            // 输出当前深度的搜索结果（调试用，Botzone 只看 stdout）
            cerr << "depth=" << depth
                 << " best=(" << bestX << "," << bestY << ")"
                 << " score=" << currentBestScore << endl;

            // 已找到必胜路径，停止加深
            if (currentBestScore >= WIN_SCORE) break;
        } else {
            // 超时了，使用上一轮完整结果（bestX, bestY 已保存）
            cerr << "depth=" << depth << " timeout, using previous result" << endl;
            break;
        }

        // 检查剩余时间是否足够搜下一层
        double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (elapsed > 0.5) {
            // 已用超过一半时间，下一层大概率搜不完，提前退出
            cerr << "time check: " << elapsed << "s, stopping deepening" << endl;
            break;
        }
    }
}

// ===================== 主函数 =====================

int main() {
    start_time = clock();
    memset(board, 0, sizeof(board));

    int n, x, y;
    cin >> n;

    // 重建棋盘
    for (int i = 0; i < n - 1; i++) {
        cin >> x >> y;
        if (x != -1) board[x][y] = 2;   // 对方
        cin >> x >> y;
        if (x != -1) board[x][y] = 1;   // 我方
    }
    cin >> x >> y;
    if (x != -1) board[x][y] = 2;

    int new_x = -1, new_y = -1;

    // ===== 开局特殊处理 =====
    if (n == 1 && x == -1) {
        // 先手第一步：下天元
        new_x = 7; new_y = 7;
    }
    else if (n == 1 && x != -1) {
        // 后手第一回合：Swap1 换手判断
        int dist = abs(x - 7) + abs(y - 7);
        if (dist <= 2) {
            new_x = -1; new_y = -1;  // 换手
        } else {
            new_x = 7; new_y = 7;
        }
    }
    else {
        // 检查棋盘是否有子
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
