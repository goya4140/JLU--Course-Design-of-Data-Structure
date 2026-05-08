/*
 * ============================================================
 *   五子棋 AI - 第四阶段：蒙特卡洛树搜索 (MCTS)
 *   适用平台：Botzone (Gomoku-Swap1)
 *   编译命令：g++ -O2 -std=c++17 -o gomoku Gomoku_v4_mcts.cpp
 * ============================================================
 *
 *   【MCTS 与 Alpha-Beta 的根本区别】
 *
 *   Alpha-Beta 依赖"估值函数"对局面打分，分数准不准决定了棋力上限。
 *   MCTS 不需要估值函数——它通过大量随机模拟（rollout）到终局，
 *   用统计胜率来判断哪步走法更好。
 *
 *   理论上，随着模拟次数趋向无穷，MCTS 会收敛到最优策略。
 *   实际中，1秒内能做 5000~20000 次模拟，已经足够形成较强的棋力。
 *
 *   【MCTS 四步循环】
 *   1. Selection（选择）：用 UCB1 公式从根往下选，平衡利用与探索
 *   2. Expansion（扩展）：在叶节点创建一个新的子节点
 *   3. Simulation（模拟）：从新节点开始随机下棋到终局
 *   4. Backpropagation（反向传播）：把结果回传更新路径上所有节点
 *
 *   【关键优化】
 *   - 模拟阶段使用"带启发的随机"而非纯随机（优先下必杀/防守点）
 *   - 候选走法限制在已有棋子周围（减少无效模拟）
 *   - 卡时控制：用 clock() 控制迭代次数，适应平台负载变化
 */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
using namespace std;

// ===================== 常量定义 =====================

const int SIZE = 15;
const int WIN_SCORE = 1000000;
const double UCB_C = 1.414;        // UCB1 探索常数（sqrt(2) 是理论最优起点）
const double TIME_LIMIT = 0.92;    // 秒，留安全余量

// 方向向量
const int DX[4] = {0, 1, 1, 1};
const int DY[4] = {1, 0, 1, -1};

int board[SIZE][SIZE];             // 0=空, 1=我方, 2=对方
clock_t start_time;

// ===================== 工具函数 =====================

inline bool inBoard(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

inline bool isTimeout() {
    return (double)(clock() - start_time) / CLOCKS_PER_SEC > TIME_LIMIT;
}

// ===================== 胜负判定 =====================

/*
 * checkWin(x, y, role)
 * 检查 (x,y) 处的 role 棋子是否形成了五连
 */
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

/*
 * checkWinOnBoard(bd, x, y, role)
 * 在指定棋盘 bd 上检查五连（用于模拟阶段，不污染全局 board）
 */
bool checkWinOnBoard(int bd[][SIZE], int x, int y, int role) {
    for (int d = 0; d < 4; d++) {
        int count = 1;
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && bd[nx][ny] == role) {
            count++; nx += DX[d]; ny += DY[d];
        }
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && bd[nx][ny] == role) {
            count++; nx -= DX[d]; ny -= DY[d];
        }
        if (count >= 5) return true;
    }
    return false;
}

// ===================== 邻近判定 =====================

bool hasNeighbor(int x, int y, int dist = 2) {
    for (int i = -dist; i <= dist; i++)
        for (int j = -dist; j <= dist; j++) {
            int nx = x + i, ny = y + j;
            if (inBoard(nx, ny) && board[nx][ny] != 0)
                return true;
        }
    return false;
}

bool hasNeighborOnBoard(int bd[][SIZE], int x, int y, int dist = 2) {
    for (int i = -dist; i <= dist; i++)
        for (int j = -dist; j <= dist; j++) {
            int nx = x + i, ny = y + j;
            if (inBoard(nx, ny) && bd[nx][ny] != 0)
                return true;
        }
    return false;
}

// ===================== 快速棋形评估（用于启发式模拟）=====================

/*
 * quickScore(bd, x, y, role)
 *
 * 快速估算在棋盘 bd 的 (x,y) 处 role 落子的价值。
 * 用于模拟阶段的启发式走法选择——不是纯随机，而是优先下"好点"。
 *
 * 【为什么模拟不能纯随机？】
 * 纯随机下完一盘棋的结果几乎没有参考价值——双方都在乱下。
 * 加入简单的启发（优先下能形成连子或防守的位置），
 * 模拟质量大幅提升，同样次数的模拟能给出更准确的胜率估计。
 */
int quickScore(int bd[][SIZE], int x, int y, int role) {
    int score = 0;
    int opponent = (role == 1) ? 2 : 1;

    for (int d = 0; d < 4; d++) {
        int count = 1, openEnds = 0;
        int nx = x + DX[d], ny = y + DY[d];
        while (inBoard(nx, ny) && bd[nx][ny] == role) {
            count++; nx += DX[d]; ny += DY[d];
        }
        if (inBoard(nx, ny) && bd[nx][ny] == 0) openEnds++;
        nx = x - DX[d]; ny = y - DY[d];
        while (inBoard(nx, ny) && bd[nx][ny] == role) {
            count++; nx -= DX[d]; ny -= DY[d];
        }
        if (inBoard(nx, ny) && bd[nx][ny] == 0) openEnds++;

        if (count >= 5) score += 100000;
        else if (count == 4 && openEnds >= 1) score += 10000;
        else if (count == 3 && openEnds == 2) score += 1000;
        else if (count == 3 && openEnds == 1) score += 100;
        else if (count == 2 && openEnds == 2) score += 10;
    }
    return score;
}

// ===================== MCTS 树节点 =====================

/*
 * MCTSNode - 蒙特卡洛搜索树的节点
 *
 * 每个节点代表一个"局面"，由从根节点到此处的走法序列确定。
 *
 * 关键属性：
 *   wins     - 经过此节点的所有模拟中，"创建此节点的那方"赢的次数
 *   visits   - 经过此节点的总模拟次数
 *   moveX/Y  - 到达此节点的那一步落子坐标
 *   role     - 下出这步棋的是谁（1=我方, 2=对方）
 *   children - 子节点列表
 *   parent   - 父节点
 *
 * 注意 wins 的归属：
 *   wins 记录的是"role 方"的胜利次数。
 *   也就是说，如果 role=1（我方下了这步），wins 就是我方在此节点下方赢的次数。
 *   UCB1 计算时，站在父节点（对方）的角度看，要用 (ni - wi)/ni 而非 wi/ni。
 */
struct MCTSNode {
    int moveX, moveY;          // 到达此节点的走法
    int role;                  // 下出这步棋的是谁
    int wins;                  // 该 role 的获胜次数
    int visits;                // 总访问次数
    MCTSNode* parent;
    vector<MCTSNode*> children;
    vector<pair<int,int>> untriedMoves;  // 尚未扩展的走法

    MCTSNode(int mx, int my, int r, MCTSNode* p)
        : moveX(mx), moveY(my), role(r), wins(0), visits(0), parent(p) {}

    ~MCTSNode() {
        for (auto c : children) delete c;
    }

    /* 是否还有未扩展的走法 */
    bool hasUntriedMoves() const {
        return !untriedMoves.empty();
    }

    /* 是否已完全展开（所有走法都已扩展） */
    bool isFullyExpanded() const {
        return untriedMoves.empty();
    }

    /* 是否是叶节点（没有子节点） */
    bool isLeaf() const {
        return children.empty();
    }

    /*
     * UCB1 选择最优子节点
     *
     * 公式：UCB1 = (wi / ni) + C × sqrt(ln(N) / ni)
     *
     * 注意：我们站在"当前节点"的角度选子节点。
     * 当前节点的 role 是"上一步下棋的人"。
     * 子节点的 role 是"这一步下棋的人"。
     *
     * 我们要选对"当前节点的 role"最有利的子节点。
     * 子节点的 wins 记录的是子节点 role 的胜利次数。
     * 对当前节点 role 来说，子节点中"对方赢得少"的更好。
     * 所以用 (ni - wi) / ni 而非 wi / ni。
     */
    MCTSNode* bestChild() {
        MCTSNode* best = nullptr;
        double bestUCB = -1e18;

        for (auto c : children) {
            // 从父节点视角：子节点的胜率应该用 (visits - wins) / visits
            // 因为子节点的 wins 是子节点 role 的胜次，而父节点 role 是对手
            double exploit = (double)(c->visits - c->wins) / (c->visits + 1e-6);
            double explore = UCB_C * sqrt(log((double)this->visits + 1) / (c->visits + 1e-6));
            double ucb = exploit + explore;

            if (ucb > bestUCB) {
                bestUCB = ucb;
                best = c;
            }
        }
        return best;
    }

    /*
     * 选择访问次数最多的子节点作为最终决策
     *
     * 为什么不选胜率最高的？因为访问次数多 = UCB1 反复选中 = 综合评价最好。
     * 胜率高但访问次数很少的节点可能只是"运气好"。
     * 这是 MCTS 的标准做法。
     */
    MCTSNode* mostVisitedChild() {
        MCTSNode* best = nullptr;
        int maxVisits = -1;
        for (auto c : children) {
            if (c->visits > maxVisits) {
                maxVisits = c->visits;
                best = c;
            }
        }
        return best;
    }
};

// ===================== 生成候选走法 =====================

/*
 * getCandidateMoves()
 * 在全局 board 上生成候选走法（已有棋子周围 2 格内的空位）
 */
vector<pair<int,int>> getCandidateMoves() {
    vector<pair<int,int>> moves;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (board[i][j] == 0 && hasNeighbor(i, j, 2))
                moves.push_back({i, j});
    return moves;
}

/*
 * getCandidateMovesOnBoard(bd)
 * 在指定棋盘上生成候选走法（用于模拟阶段）
 */
vector<pair<int,int>> getCandidateMovesOnBoard(int bd[][SIZE]) {
    vector<pair<int,int>> moves;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (bd[i][j] == 0 && hasNeighborOnBoard(bd, i, j, 2))
                moves.push_back({i, j});
    return moves;
}

// ===================== MCTS 核心流程 =====================

/*
 * 第三步：模拟（Simulation / Rollout）
 *
 * 从当前局面开始，双方交替下棋直到分出胜负或达到步数上限。
 *
 * 【启发式模拟策略】（不是纯随机！）
 * 1. 如果当前方有一步能直接赢 → 下那步（必杀）
 * 2. 如果对方有一步能直接赢 → 堵那步（必守）
 * 3. 否则，按 quickScore 加权随机选择（好点更容易被选中）
 *
 * 参数：
 *   simBoard  - 模拟用的棋盘副本（不影响全局 board）
 *   startRole - 当前轮到谁走（1=我方, 2=对方）
 *   lastX/Y   - 上一步的坐标（用于快速检查上一步是否获胜）
 *
 * 返回：
 *   1 = 我方赢, 2 = 对方赢, 0 = 平局
 */
int simulate(int simBoard[][SIZE], int startRole, int lastX, int lastY) {
    const int MAX_SIM_STEPS = 60;  // 模拟最多走60步（避免模拟太久）
    int currentRole = startRole;

    for (int step = 0; step < MAX_SIM_STEPS; step++) {
        // 生成候选走法
        vector<pair<int,int>> moves = getCandidateMovesOnBoard(simBoard);
        if (moves.empty()) return 0;  // 没有空位，平局

        int opponent = (currentRole == 1) ? 2 : 1;
        int bestX = -1, bestY = -1;

        // ===== 启发式 1：检查是否有直接获胜的走法 =====
        for (auto& [mx, my] : moves) {
            simBoard[mx][my] = currentRole;
            if (checkWinOnBoard(simBoard, mx, my, currentRole)) {
                simBoard[mx][my] = 0;
                return currentRole;  // 当前方直接赢
            }
            simBoard[mx][my] = 0;
        }

        // ===== 启发式 2：检查是否需要堵对方的必胜点 =====
        for (auto& [mx, my] : moves) {
            simBoard[mx][my] = opponent;
            if (checkWinOnBoard(simBoard, mx, my, opponent)) {
                simBoard[mx][my] = 0;
                bestX = mx; bestY = my;  // 必须堵这里
                break;
            }
            simBoard[mx][my] = 0;
        }

        // ===== 启发式 3：加权随机选择 =====
        if (bestX == -1) {
            // 计算每个候选点的得分
            vector<int> scores(moves.size());
            int totalScore = 0;
            for (int i = 0; i < (int)moves.size(); i++) {
                auto& [mx, my] = moves[i];
                int s = quickScore(simBoard, mx, my, currentRole)
                      + quickScore(simBoard, mx, my, opponent);
                scores[i] = s + 1;  // +1 保证最低分也有概率被选
                totalScore += scores[i];
            }

            // 按分数加权随机选一个（分高的被选概率大）
            int r = rand() % totalScore;
            int cumulative = 0;
            for (int i = 0; i < (int)moves.size(); i++) {
                cumulative += scores[i];
                if (cumulative > r) {
                    bestX = moves[i].first;
                    bestY = moves[i].second;
                    break;
                }
            }
        }

        // 落子
        if (bestX == -1) {
            // 兜底：随机选一个
            int idx = rand() % moves.size();
            bestX = moves[idx].first;
            bestY = moves[idx].second;
        }
        simBoard[bestX][bestY] = currentRole;

        // 检查是否获胜
        if (checkWinOnBoard(simBoard, bestX, bestY, currentRole)) {
            return currentRole;
        }

        // 切换到对方
        currentRole = opponent;
    }

    return 0;  // 步数用完，平局
}

/*
 * MCTS 主循环
 *
 * 参数：rootRole - 当前轮到谁走
 * 返回：最佳走法的坐标 (bestX, bestY)
 *
 * 流程：
 *   1. 创建根节点
 *   2. 循环直到时间用完：
 *      a. Selection：从根往下用 UCB1 选择
 *      b. Expansion：在叶节点扩展一个新子节点
 *      c. Simulation：从新节点开始随机模拟
 *      d. Backpropagation：回传结果
 *   3. 选择根节点下访问次数最多的子节点
 */
void mctsSearch(int& bestX, int& bestY) {
    int rootRole = 1;  // 当前轮到我方走

    // 创建根节点
    // 根节点的 role=2（表示"上一步是对方走的"，现在轮到我方）
    MCTSNode* root = new MCTSNode(-1, -1, 2, nullptr);
    root->untriedMoves = getCandidateMoves();

    // 如果没有候选走法（不应该发生）
    if (root->untriedMoves.empty()) {
        bestX = 7; bestY = 7;
        delete root;
        return;
    }

    int iterations = 0;

    // ===== MCTS 主循环 =====
    while (!isTimeout()) {
        iterations++;

        // 复制棋盘用于本次迭代（每次迭代独立，不污染全局 board）
        int simBoard[SIZE][SIZE];
        memcpy(simBoard, board, sizeof(board));

        MCTSNode* node = root;
        int currentRole = rootRole;  // 当前轮到谁走
        int winner = 0;              // 本次迭代的最终赢家
        bool terminated = false;     // 是否在选择/扩展阶段就已分出胜负

        // ===== 第一步：选择（Selection）=====
        // 从根节点往下，如果节点已完全展开且有子节点，用 UCB1 选择最优子节点
        while (node->isFullyExpanded() && !node->isLeaf()) {
            node = node->bestChild();
            // 在模拟棋盘上执行这步走法
            simBoard[node->moveX][node->moveY] = currentRole;

            // 检查是否已经有人赢了
            if (checkWinOnBoard(simBoard, node->moveX, node->moveY, currentRole)) {
                winner = currentRole;
                terminated = true;
                break;
            }

            currentRole = (currentRole == 1) ? 2 : 1;
        }

        // ===== 第二步：扩展（Expansion）=====
        // 如果当前节点还有未尝试的走法，且尚未分出胜负
        if (!terminated && node->hasUntriedMoves()) {
            // 随机选一个未尝试的走法
            int idx = rand() % node->untriedMoves.size();
            auto [mx, my] = node->untriedMoves[idx];

            // 从未尝试列表中移除（用末尾元素覆盖，O(1)删除）
            node->untriedMoves[idx] = node->untriedMoves.back();
            node->untriedMoves.pop_back();

            // 在模拟棋盘上落子
            simBoard[mx][my] = currentRole;

            // 创建新的子节点
            MCTSNode* child = new MCTSNode(mx, my, currentRole, node);
            node->children.push_back(child);
            node = child;

            // 检查扩展后是否已分胜负
            if (checkWinOnBoard(simBoard, mx, my, currentRole)) {
                winner = currentRole;
                terminated = true;
            } else {
                // 为新节点生成未尝试走法列表
                int nextRole = (currentRole == 1) ? 2 : 1;
                node->untriedMoves = getCandidateMovesOnBoard(simBoard);
                currentRole = nextRole;
            }
        }

        // ===== 第三步：模拟（Simulation）=====
        // 只有在选择/扩展阶段没有分出胜负时才需要模拟
        if (!terminated) {
            winner = simulate(simBoard, currentRole, -1, -1);
        }

        // ===== 第四步：反向传播（Backpropagation）=====
        // 从当前节点回溯到根，更新每个节点的 wins 和 visits
        MCTSNode* backNode = node;
        while (backNode != nullptr) {
            backNode->visits++;
            // 如果赢家是该节点的 role，则 wins++
            if (winner == backNode->role) {
                backNode->wins++;
            }
            backNode = backNode->parent;
        }
    }

    // ===== 选择最终走法 =====
    MCTSNode* bestChild = root->mostVisitedChild();

    if (bestChild != nullptr) {
        bestX = bestChild->moveX;
        bestY = bestChild->moveY;
    } else {
        bestX = root->untriedMoves[0].first;
        bestY = root->untriedMoves[0].second;
    }

    // 调试信息
    cerr << "MCTS iterations: " << iterations << endl;
    cerr << "Best move: (" << bestX << "," << bestY << ")"
         << " visits=" << (bestChild ? bestChild->visits : 0)
         << " winrate=" << (bestChild && bestChild->visits > 0 ?
            (double)(bestChild->visits - bestChild->wins) / bestChild->visits * 100 : 0)
         << "%" << endl;

    // 输出所有子节点的统计（调试用）
    for (auto c : root->children) {
        cerr << "  (" << c->moveX << "," << c->moveY << ")"
             << " visits=" << c->visits
             << " wins=" << c->wins
             << " wr=" << (c->visits > 0 ?
                (double)(c->visits - c->wins) / c->visits * 100 : 0)
             << "%" << endl;
    }

    delete root;
}

// ===================== 顶层决策 =====================

/*
 * findBestMove(bestX, bestY)
 *
 * 在启动 MCTS 搜索之前，先做快速检测：
 * 1. 我方能一步赢 → 直接下
 * 2. 对方能一步赢 → 必须堵
 * 这两种情况不需要花时间搜索。
 */
void findBestMove(int& bestX, int& bestY) {
    vector<pair<int,int>> moves = getCandidateMoves();

    if (moves.empty()) {
        bestX = 7; bestY = 7;
        return;
    }

    // 快速检测：我方直接赢
    for (auto& [mx, my] : moves) {
        board[mx][my] = 1;
        if (checkWin(mx, my, 1)) {
            board[mx][my] = 0;
            bestX = mx; bestY = my;
            return;
        }
        board[mx][my] = 0;
    }

    // 快速检测：堵对方必杀
    for (auto& [mx, my] : moves) {
        board[mx][my] = 2;
        if (checkWin(mx, my, 2)) {
            board[mx][my] = 0;
            bestX = mx; bestY = my;
            return;
        }
        board[mx][my] = 0;
    }

    // MCTS 搜索
    mctsSearch(bestX, bestY);
}

// ===================== 主函数 =====================

int main() {
    start_time = clock();
    srand(start_time);   // 随机数种子
    memset(board, 0, sizeof(board));

    int n, x, y;
    cin >> n;

    // 重建棋盘
    for (int i = 0; i < n - 1; i++) {
        cin >> x >> y;
        if (x != -1) board[x][y] = 2;
        cin >> x >> y;
        if (x != -1) board[x][y] = 1;
    }
    cin >> x >> y;
    if (x != -1) board[x][y] = 2;

    int new_x = -1, new_y = -1;

    // ===== 开局特殊处理 =====
    if (n == 1 && x == -1) {
        new_x = 7; new_y = 7;
    }
    else if (n == 1 && x != -1) {
        int dist = abs(x - 7) + abs(y - 7);
        if (dist <= 2) {
            new_x = -1; new_y = -1;
        } else {
            new_x = 7; new_y = 7;
        }
    }
    else {
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
