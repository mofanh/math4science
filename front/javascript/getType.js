var trees = new Array("redwood", "bay", "cedar", "oak", "maple");
delete trees[3];
// trees[3] = undefined;
if (3 in trees) {
  // 不会被执行
  console.log("124");
}

// console.log(typeof trees.length);
console.log(trees instanceof Array);