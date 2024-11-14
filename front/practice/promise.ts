let promise = new Promise((resolve, reject) => {
  console.log(1);
  //   resolve(1);
  throw new Error("Something went wrong");
});

promise
  .then((result) => {
    console.log(2);
  })
  .catch((error) => {
    console.log(3);
  })
  .then((result) => {
    console.log(2.1);
  })
  .finally(() => {
    console.log(4);
  })
  .then(() => {
    console.log(5);
  });
