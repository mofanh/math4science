// 题目： 给定一个数组 nums 和一个目标值 target，在该数组中找出和为目标值的两个数
// 输入： nums: [8, 2, 6, 5, 4, 1, 3] ； target:7
// 输出： [2, 5]

// const nums = [8, 2, 6, 5, 4, 1, 3];
// const target = 7;

// function find(nums, target) {
//   let aMap = new Map();

//   for (let i = 0; i < nums.length; i++) {
//     if (nums[i] >= target) {
//       continue;
//     }
//     if (aMap.get(nums[i])) {
//       return [nums[i], aMap.get(nums[i])];
//     }
//     aMap.set(target - nums[i], nums[i]);
//   }
// }

// console.log(find(nums, target));

// 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height) 。
// 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
// 返回容器可以储存的最大水量。
// 输入：[1,8,6,2,5,4,8,3,7]
// 输出：49
// 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49

const nums = [1, 8, 6, 2, 5, 4, 8, 3, 7];

function findMax(nums) {
  let res = 0;
  function dfs(l, r) {
    if (l > r || r >= nums.length) {
      return;
    }
    const cur = (r - l) * Math.min(nums[l], nums[r]);
    if (cur > res) {
      res = cur;
    }
    dfs(l, r + 1);
    dfs(l + 1, r);
  }

  dfs(0, 1);

  return res;
}
console.log(findMax(nums));

// for(let i = 0; i < nums.length; i++){
//     for(let j = 0; j < nums.length; j++){

//     }
// }
