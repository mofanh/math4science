const promise = new Promise((resolve, reject) => {
  console.log("Promise 执行函数");
  resolve();
}).then((result) => {
  console.log("Promise 回调（.then）");
});

setTimeout(() => {
  console.log("新一轮事件循环：Promise（已完成）", promise);
}, 0);

console.log("Promise（队列中）", promise);
