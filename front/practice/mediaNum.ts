const nums = [1,1, 1, 2, 2, 3, 3, 3];

function findNums(nums) {
  let quee = [];
  for (let i = 0; i < nums.length; i++) {
    let ptr = 0;
    if (quee.length === 0) {
      const temp = [nums[i]];
      quee.push(temp);
      continue;
    }

    let isChange = false;
    while (ptr < quee.length) {
      if (quee[ptr][quee[ptr].length - 1] === nums[i] - 1) {
        quee[ptr].push(nums[i]);
        isChange = true;
        break;
      }
      ptr++;
    }

    if (!isChange) {
      const temp = [nums[i]];
      quee.push(temp);
    }
  }

  console.log(quee);
}

// findNums(nums);

function findNums2(nums){
    let que = [[nums[0]]]
    for(let num = 1; num < nums.length; num++){
        let isInsert = false;
        for(let i = 0; i < que.length; i++){
            if(que[i].indexOf(nums[num]) === -1 && que[i].indexOf(nums[num]-1) !== -1){
                que[i].push(nums[num]);
                isInsert = true;
                break;
            }
        }
        if(!isInsert){
            que.push([nums[num]]);
        }
    }

    console.log(que)
}

findNums2(nums);