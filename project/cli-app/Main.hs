module Main (main) where

import Options.Applicative
import Lib

parsingArgument :: Parser Int
parsingArgument = argument auto (metavar "<number of iterations>")

main :: IO ()
main = entryPoint =<< execParser opts
    where
        opts = info (parsingArgument <**> helper)
            ( fullDesc
            <> progDesc "This application finds the solution of the children's farmer problem"
            <> header "Sample CLI application written in Haskell" )
