function jl(func, delay = 1000) {
  let oldTime = new Date();

  return function (...args) {
    let newTime = new Date();
    if (newTime - oldTime >= delay) {
      func.apply(args);
      oldTime = newTime;
    }
  };
}

let aaa = jl(() => {
  console.log(23);
});

aaa();
setTimeout(() => {
  console.log(12333);
}, 2000);
aaa();
aaa();
