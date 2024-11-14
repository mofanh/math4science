function generate(numRows) {
    let _i = 0
    let dp = Array(numRows).fill(0).map(() => {
        _i++;
        return Array(_i).fill(0);
    })
    for(let i = 0; i < dp.length; i++){
        dp[i][0] = 1;
        dp[i][dp[i].length-1] = 1;
    }

    for(let i = 2; i < dp.length; i++){
        for(let j = 1; j < dp[i].length; j++){
            if(j === dp[i].length-1){
                continue
            }
            dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
        }
    }

    return dp;
};

console.log(generate(6))