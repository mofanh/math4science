// 被触发后刷新计时时间、并在计时结束后调用函数

function fd(fn, wait) {
  let timeout;
  return function () {
    const args = arguments;

    clearTimeout(timeout);
    timeout = setTimeout(fn(args), wait);
  };
}

let fn = () => {
  console.log("1");
};

let fd1 = fd(fn, 100);
fd1;
