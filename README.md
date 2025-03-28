# llama2-haskell-inference

Haskell version of llama2.c **(Deprecated, please see: https://github.com/chris-ch/llama2.hs)**

## Pre-requisites

You will need to install a few training sets,
for example the mini stories from [llama.c](https://github.com/karpathy/llama2.c#models).

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

There is also a bigger version, for better stories:

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

### Using nix

Make sure flakes are enabled: add `experimental-features = nix-command flakes` to `~/.config/nix/nix.conf`

```shell
nix develop
code .
```

Make sure HLS is set in `.vscode/settings.json` in order to get error messages in VSCode:

```json
{ "haskell.serverExecutablePath": "haskell-language-server" }
```

### Running llama2 using cabal

```shell
cabal update
```

```shell
cabal run -- llama2 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

For testing purposes, you can set the _seed_ option to some value to always get the same output:

```shell
cabal run -- llama2 --seed 1 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

Generated output for that particular seed:

```text
<s>
In that little town, there was a girl named Jane. Jane had a big, colorful folder. She loved her folder very much. She took it everywhere she went.
One day, Jane went to the park with her folder. She saw a boy named Tim. Tim was sad. Jane asked, "Why are you sad, Tim?" Tim said, "I lost my toy car." Jane wanted to help Tim. She said, "Let's look for your toy car together."
They looked and looked. Then, they found the toy car under a tree. Tim was very happy. He said, "Thank you, Jane!" Jane felt good because she helped her friend. The moral of the story is that helping others can make you feel good too.
<s>
```

### Unit testing

Running tests matching specifically "FFN":

```shell
HSPEC_COLOR=yes cabal test --test-show-details="streaming" --keep-going --test-option=--match --test-option="FFN"
```
