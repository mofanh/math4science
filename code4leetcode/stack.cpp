#include<stdio.h>
#include<iostream>
#include"node.cpp"

class stack
{
private:
    node *head;
    // node *end;
public:
    stack(/* args */);
    ~stack();
    void push(int in);
    int pop();
};

stack::stack(/* args */)
{
    head = new node();
    // end = head;
}

void stack::push(int in)
{
    node *temp = new node;
    temp->val = in;
    temp->next = head->next;
    head->next = temp;
}

int stack::pop()
{
    if(head->next == NULL)
    {
        return -1;
    }
    node *temp = head->next;
    int out = temp->val;
    head->next = temp->next;
    delete temp;

    return out;
}

stack::~stack()
{
    while (head->next != NULL)
    {
        pop();
    }
    delete head; // 释放栈顶节点
}
