function fd(func, wait) {
  let timeout;

  return function () {
    let context = this;
    let args = arguments;
    ClearTimeout(timeout);
    timeout = setTimeout(function () {
      func.apply(context, args);
    }, wait);
  };
}

const f1 = fd(console.log(123), 1000);
