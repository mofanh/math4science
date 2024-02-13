#include <iostream>
#include <queue>
#include <deque>
#include <vector>

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

    std::vector<int> vec3 = {1, 2, 3, 4, 5};
    std::vector<int>* vec_ptr = &vec3; // 创建一个指向vec3的指针

    return 0;
}