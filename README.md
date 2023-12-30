# llama2-haskell-inference
Haskell version of llama2.c

## Pre-requisites
You will need to install a few training sets,
for example the mini stories from [llama.c](https://github.com/karpathy/llama2.c#models).

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

### Running llama2 using stack
```shell
stack run -- cli-app --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

For testing purposes, you can set the _seed_ option to some value to always get the same output:

```shell
stack run -- cli-app --seed 1 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

Generated output for that particular seed:
TBD.

### Debugging / Profiling
```shell
stack build --profile  --executable-profiling --library-profiling
stack run --profile -- cli-app --seed 1 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town" +RTS -p
```


### Unit testing
Running tests matching specifically "Helper":

```shell
stack test --test-arguments='--match "Helper"'
```

