class MyPromise {
  constructor(executor) {
    this.executor = executor;
    this.state = undefined;
    this.status = "pending";

    const resolved = (val) => {
      if (this.status === "pending") {
        this.state = val;
        this.status = "resolved";
      }
    };

    const rejected = (val) => {
      if (this.status === "pending") {
        this.state = val;
        this.status = "rejected";
      }
    };

    this.executor(resolved, rejected);
  }

//   then(onFulfilled, onRejected) {

//   }
then(onFulfilled, onRejected) {
    const self = this
    if (this.status === 'pending') {
      /**
       * 当 promise 的状态仍然处于 ‘pending’ 状态时，需要将注册 onFulfilled、onRejected 方法放到 promise 的 onFulfilledFunctions、onRejectedFunctions 中备用
       */
      return new MyPromise((resolve, reject) => {
        this.onFulfilledFunctions.push(() => {
          const thenReturn = onFulfilled(self.value)
          resolve(thenReturn)
        })
        this.onRejectedFunctions.push(() => {
          const thenReturn = onRejected(self.value)
          resolve(thenReturn)
        })
      })
    } else if (this.status === 'fulfilled') {
      return new MyPromise((resolve, reject) => {
        const thenReturn = onFulfilled(self.value)
        resolve(thenReturn)
      })
    } else {
      return new MyPromise((resolve, reject) => {
        const thenReturn = onRejected(self.value)
        resolve(thenReturn)
      })
    }
  }
}
