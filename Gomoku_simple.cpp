#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
using namespace std;

const int SIZE = 15;
int board[SIZE][SIZE] = { 0 };

int main()
{
	int x, y, n;

	cin >> n;
	for (int i = 0; i < n - 1; i++) {
		cin >> x >> y; if (x != -1) board[x][y] = -1;	
		cin >> x >> y; if (x != -1) board[x][y] = 1;	
	}
	cin >> x >> y;
	if (x != -1) board[x][y] = -1;	//�Է�

	int new_x, new_y;
	if (x != -1 && n == 1) {  
		new_x = -1; new_y = -1;
	} else {  
		int avail_x[SIZE*SIZE], avail_y[SIZE*SIZE]; 
		int cnt = 0;
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				if (board[i][j] == 0) {
					avail_x[cnt] = i;
					avail_y[cnt] = j;
					cnt++;
				}
		srand(time(0));
		int rand_pos = rand() % cnt;
		new_x = avail_x[rand_pos];
		new_y = avail_y[rand_pos];

	}

	printf("%d %d\n", new_x, new_y);
	return 0;
}