function longestCommonSubsequence(text1, text2) {
    const n = text1.length;
    const m = text2.length;
    let dp = Array(n+1).fill(0).map(() => Array(m+1).fill(0));

    // for(let i = 0; i < dp.length; i++){
    //     dp[i][0] = 1;
    // }
    // for(let i = 0; i < dp[0].length; i++){
    //     dp[0][i] = 1;
    // }
    // dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1])
    for(let i = 1; i < dp.length; i++){
        for(let j = 1; j < dp[0].length; j++){
            if(text1[i] === text2[j]){
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[n][m];
};

const text1 = "abcde"
const text2 = "ace"

console.log(longestCommonSubsequence(text1, text2))