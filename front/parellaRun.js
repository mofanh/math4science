const promise1 = Promise.resolve(1);
const promise2 = new Promise((resolve, reject) => {
  setTimeout(resolve, 2000, 2);
});
const promise3 = new Promise((resolve, reject) => {
  setTimeout(resolve, 1000, 3);
});

Promise.all([promise1, promise2, promise3]).then((values) => {
  console.log(values);
});

setTimeout(() => console.log(1), 1000);
setTimeout(() => console.log(2), 2000);
// Expected output: Array [3, 42, "foo"]
