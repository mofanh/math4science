#include<stdio.h>
#include<iostream>
// #include"node.cpp"
#include"stack.cpp"


int main()
{
    printf("hello world\n");

    stack s;
    s.push(1);
    s.push(2);
    std::cout << s.pop() << std::endl; // 输出 2
    std::cout << s.pop() << std::endl; // 输出 1

    return 0;
}