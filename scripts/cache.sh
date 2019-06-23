cache_dir="data/cache"

if [ $2 ];then
    cache_dir=$cache_dir"_"$2"/"
else
    cache_dir=$cache_dir"/"
fi

if [ "$1" = "clear" ];then
    rm -r $cache_dir"*"
    mkdir $cache_dir"data_loader/"
elif [ "$1" = "store" ];then
    mv "data/cache/" $cache_dir
    mkdir "data/cache/"
    mkdir "data/cache/data_loader/"
elif [ "$1" = "restore" ];then
    rm -r "data/cache/"
    mv $cache_dir "data/cache/"
else
    echo "unexpected command $1"
    exit 1
fi
