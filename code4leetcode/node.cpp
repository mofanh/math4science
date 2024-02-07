#pragma once
#include<stdio.h>
#include<iostream>

class node
{
private:

public:
    int val;
    node *next;
    node(/* args */);
    ~node();
};

node::node(/* args */):next(nullptr)
{
}

node::~node()
{
}
