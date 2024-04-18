class Scheduler {
  list = [];
  maxNum = 2;
  curNum = 0;

  add(promiseCreator) {
    this.list.push(promiseCreator);
  }

  start() {
    for (let i = 0; i < this.maxNum; i++) {
      this.doNext();
    }
  }

  doNext() {
    if (this.list.length && this.curNum < this.maxNum) {
      this.curNum++;
      this.list
        .shift()()
        .then(() => {
          this.curNum--;
          this.doNext();
        });
    }
  }
}

const timeout = (time) => new Promise((resolve) => setTimeout(resolve, time));

const scheduler = new Scheduler();

let addTask = (order, time) => {
  scheduler.add(() => {
    return timeout(time).then(() => {
      console.log(order);
    });
  });
};

addTask(1, 1000);
addTask(2, 500);
addTask(3, 300);
addTask(4, 400);

scheduler.start();
