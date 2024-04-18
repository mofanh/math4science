const timeout = (time) => new Promise((resolve) => setTimeout(resolve, time));

let p1 = timeout(1000).then(() => {
  console.log(1);
  return 1;
});
let p2 = timeout(2000).then(() => {
  console.log(2);
  return 2;
});
let p3 = timeout(3000).then(() => {
  console.log(3);
  return 13;
});

Promise.all([p1, p2, p3]).then((values) => {
  console.log(values);
});
