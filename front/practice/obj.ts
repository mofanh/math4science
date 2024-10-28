function maxSubArraySum(arr) {
    let currentMax = 0;
    let globalMax = arr[0] || 0; // 如果数组全为负数，则取第一个元素作为初始值
  
    for (let i = 0; i < arr.length; i++) {
      // 更新当前元素到末尾的最大和
      currentMax = Math.max(arr[i], currentMax + arr[i]);
      // 如果当前元素到末尾的最大和大于全局最大和，则更新全局最大和
      if (currentMax > globalMax) {
        globalMax = currentMax;
      }
    }
  
    return globalMax;
  }
  
  // 示例
  const arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
  console.log(maxSubArraySum(arr)); // 输出：6，因为[4, -1, 2, 1]是最大和的连续子数组
  `