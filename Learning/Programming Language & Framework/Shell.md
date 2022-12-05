## 测试是否是可执行文件
用`test -x`选项，例子：
```bash
executable_dry="dryrun1.out"
if test -x $executable_dry ; then
    ./$executable_dry
    OP_MODE=$?
else
    printf "===== Increase executable of dryrun1.out =====\n"
    chmod +x $executable_dry
    ./$executable_dry
    OP_MODE=$?
fi
```


## $的用法
```txt
$0：Shell 的命令本身
$1 到 $9：表示 Shell 的第几个参数
$?：显示最后命令的执行情况(上条命令的返回值。0：表示没有错误，其他任何数值：表示有错误。)
$#：传递到脚本的参数个数
$$：脚本运行的当前进程 ID 号
$*：以一个单字符串显示所有向脚本传递的参数
$!：后台运行的最后一个进程的 ID 号
$-：显示 Shell 使用的当前选项
```

使用 ${#} 获取变量字符串长度：
```shell
[root@localhost etc]# s=helloworld
[root@localhost etc]# echo "s.length = ${#s}"
s.length = 10
```
单引号 ‘’ 括起来的字符串不会进行插值，并使用 $# 获取脚本或函数参数的个数：
```shell
[root@localhost ~]# echo 'echo $#' > ping.sh
[root@localhost ~]# sh ping.sh 1 2 3
3
```

更多用法参考[https://blog.csdn.net/jake_tian/article/details/97274630>


## timeout命令的用法
用timeout --help查看用法
timeout可以将跟在后面的command运行时间和规定时间进行对比,如果超时则自动kill掉该命令，并返回状态号124
非常适合用在反复执行同一个.sh、.py等（并且不希望因为其中某一轮循环超时挂起卡住而导致阻塞在这里循环跑不满）的场景
技巧：
可以在timeout语句的下一条写`$?`是否等于124的判断 (`$?`是获取上一条语句的状态号)
等于124则发生了超时，此时可以写个error的log输出作为该轮循环的执行错误记录，如下
```SHELL
for i in $(seq 1 5000)
do
		# timeout 5s ./$executable_file 1>stdout.txt 2>stderr.txt || [ $? -eq 124 ] && echo timeouted
    timeout 5s ./$executable_file 1>stdout.txt 2>stderr.txt
    if [[ $? == "124" ]]
    then
        echo timeouted_____________________________________________________________________________________________________________________________________________
        echo $i,DUE,hang>> DUE.txt
        echo $i,DUE,hang>> outcome.txt
        echo ,5s>> basic.txt
    else
        printf "===== Create out.txt =====\n"
        # ./$executable_file
        printf "===== Create diff.log =====\n"
        diff out.txt golden/out.txt>diff.log
        # after injection
        python parse_diff.py $OP_MODE $i
    fi
done
```