new Promise((resolve, reject) => {
  console.log("初始化");

  resolve();
})
  .then(() => {
    throw new Error("有哪里不对了");

    console.log("执行「这个」");
  })
  .catch(() => {
    console.log("执行「那个」");
  })
  .then(() => {
    console.log("执行「这个」，无论前面发生了什么");
  });

// doSomething()
//   .then(function (result) {
//     // 如果使用完整的函数表达式：返回 Promise
//     return doSomethingElse(result);
//   })
//   // 如果使用箭头函数：省略大括号并隐式返回结果
//   .then((newResult) => doThirdThing(newResult))
//   // 即便上一个 Promise 返回了一个结果，后一个 Promise 也不一定非要使用它。
//   // 你可以传入一个不使用前一个结果的处理程序。
//   .then((/* 忽略上一个结果 */) => doFourthThing())
//   // 总是使用 catch 终止 Promise 链，以保证任何未处理的拒绝事件都能被捕获！
//   .catch((error) => console.error(error));
