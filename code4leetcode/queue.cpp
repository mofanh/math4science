#include <iostream>
#include <queue>
#include <deque>

int main() {
    // 使用std::deque作为底层容器的std::queue
    std::queue<int> q;

    // 入队（头进）
    q.push(1);
    q.push(2);
    q.push(3);

    // 出队（尾出）
    while (!q.empty()) {
        int front = q.front(); // 获取队头元素
        std::cout << "Popping: " << front << std::endl;
        q.pop(); // 移除队头元素
    }

    return 0;
}