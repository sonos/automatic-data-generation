# GRID

## Install
```
python setup.py install
```

## Usage
The file `example.py` is a minimal script that can be used with `grid`.

```
python -m grid results "python example.py" --a 1 2 3 --b 5 6 7
```
All the combination of (1, 2, 3) and (5, 6, 7) are executed in simultaneously
```
[a=1 b=5] python example.py --pickle results/07405.pkl --a 1 --b 5
[a=1 b=6] python example.py --pickle results/04524.pkl --a 1 --b 6
[a=1 b=7] python example.py --pickle results/08265.pkl --a 1 --b 7
[a=2 b=5] python example.py --pickle results/00672.pkl --a 2 --b 5
[a=2 b=6] python example.py --pickle results/06681.pkl --a 2 --b 6
[a=2 b=7] python example.py --pickle results/06732.pkl --a 2 --b 7
[a=3 b=5] python example.py --pickle results/02937.pkl --a 3 --b 5
[a=3 b=6] python example.py --pickle results/08754.pkl --a 3 --b 6
[a=3 b=7] python example.py --pickle results/08521.pkl --a 3 --b 7
[a=2 b=5] computation 1 / 3
[a=2 b=6] computation 1 / 3
[a=1 b=7] computation 1 / 3
[a=1 b=6] computation 1 / 3
[a=1 b=5] computation 1 / 3
[a=2 b=7] computation 1 / 3
[a=3 b=6] computation 1 / 3
[a=3 b=7] computation 1 / 3
[a=3 b=5] computation 1 / 3
[a=2 b=5] computation 2 / 3
[a=2 b=6] computation 2 / 3
[a=1 b=7] computation 2 / 3
[a=1 b=6] computation 2 / 3
[a=1 b=5] computation 2 / 3
[a=2 b=7] computation 2 / 3
[a=3 b=6] computation 2 / 3
[a=3 b=7] computation 2 / 3
[a=3 b=5] computation 2 / 3
[a=2 b=5] computation 3 / 3
[a=2 b=6] computation 3 / 3
[a=1 b=7] computation 3 / 3
[a=1 b=6] computation 3 / 3
[a=1 b=5] computation 3 / 3
[a=2 b=7] computation 3 / 3
[a=3 b=5] computation 3 / 3
[a=3 b=6] computation 3 / 3
[a=3 b=7] computation 3 / 3
```

```
python -m grid results "python example.py" --a 1 2 3 --b 5 6 7 8
```
The results already done are not executed again
```
[a=1 b=5] already done
[a=1 b=6] already done
[a=1 b=7] already done
[a=1 b=8] python example.py --pickle results/07530.pkl --a 1 --b 8
[a=2 b=5] already done
[a=2 b=6] already done
[a=2 b=7] already done
[a=2 b=8] python example.py --pickle results/03495.pkl --a 2 --b 8
[a=3 b=5] already done
[a=3 b=6] already done
[a=3 b=7] already done
[a=3 b=8] python example.py --pickle results/06283.pkl --a 3 --b 8
[a=2 b=8] computation 1 / 3
[a=3 b=8] computation 1 / 3
[a=1 b=8] computation 1 / 3
[a=2 b=8] computation 2 / 3
[a=3 b=8] computation 2 / 3
[a=1 b=8] computation 2 / 3
[a=3 b=8] computation 3 / 3
[a=2 b=8] computation 3 / 3
[a=1 b=8] computation 3 / 3
```

Example: using slurm and only varying parameter b
```
python -m grid results "srun --partition gpu --qos gpu --gres gpu:1 --time 3-00:00:00 --mem 12G --pty python example.py --a 4" --b:int 5 6 7
```

## Subcommands

- `pyhton -m grid.clear log_dir` remove files with no content (typically resultant of interrupted runs)
- `pyhton -m grid.merge log_dir_src log_dir_dst`
- `pyhton -m grid.info log_dir`
