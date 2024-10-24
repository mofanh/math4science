#include <iostream>
#include <vector>

// 打印数组的函数
void printArray(const std::vector<int>& arr) {
    for (int i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// 交换两个元素的值
void swap(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
}

// 快速排序的分区函数，选择数组的最后一个元素作为基准值
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // 选择最后一个元素作为基准值
    int i = (low - 1); // 指向比基准值小的元素的最后一个位置

    for (int j = low; j <= high - 1; j++) {
        // 如果当前元素小于或等于基准值
        if (arr[j] <= pivot) {
            i++; // 移动指向比基准值小的元素的位置
            swap(arr[i], arr[j]);
            printArray(arr);
        }
    }
    swap(arr[i + 1], arr[high]); // 把基准值放到正确的位置
    return (i + 1);
}

// 快速排序函数
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        // pi 是分区索引，arr[pi] 现在在正确的位置
        int pi = partition(arr, low, high);

        // 分别递归地对基准值左右两边的子数组进行快速排序
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}


// 主函数
int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();

    std::cout << "原始数组: ";
    printArray(arr);

    quickSort(arr, 0, n - 1);

    std::cout << "排序后的数组: ";
    printArray(arr);

    return 0;
}