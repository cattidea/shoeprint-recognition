import argparse

if __name__ == "__main__":
    """ 主函数，解析参数并启动 """
    parser = argparse.ArgumentParser(description='shoeprint recognition')
    parser.add_argument('action', choices=['train', 'test', 'docs'],help='action type (train/test)')
    parser.add_argument('--resume', action='store_true', help='恢复已有模型继续训练')
    parser.add_argument('--no-gpu', action='store_true', help='不使用 GPU')
    parser.add_argument('--use-cache', action='store_true',
                        help='使用已有的 sample cache')

    args = parser.parse_args()

    train_config = {
        "resume": args.resume,
        "use_GPU": not args.no_gpu,
    }

    test_config = {
        "use_GPU": not args.no_gpu,
        "use_cache": args.use_cache,
    }

    if args.action == "train":
        from trainer.train import train
        train(train_config)
    elif args.action == "test":
        from infer.test import test
        test(test_config)
    elif args.action == "docs":
        from docs import docs_dev
        docs_dev()
    else:
        print("can not parse arg {}".format(args.action))
