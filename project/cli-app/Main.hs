{-# LANGUAGE RecordWildCards #-}
module Main (main) where

import Options.Applicative
    ( optional,
      (<**>),
      auto,
      fullDesc,
      help,
      info,
      long,
      metavar,
      option,
      strArgument,
      strOption,
      value,
      execParser,
      helper,
      Parser )
import System.IO ( hClose, openFile, IOMode(ReadMode) )
import qualified Data.ByteString.Lazy as BS
import Inference (run)
import System.Directory ( canonicalizePath )
import Text.Printf (printf)


data Options = Options
    { seed :: Maybe Int
    , tokenizerFile :: FilePath
    , modelFile :: FilePath
    , temperature :: Double
    , steps :: Int
    , prompt :: Maybe String
    }

-- Parser for command-line options
optionsParser :: Parser Options
optionsParser = Options
    <$> optional (option auto (long "seed" <> help "Seed for debugging"))
    <*> strOption (long "tokenizer-file" <> value "./data/tokenizer.bin" <> help "Tokenizer binary file")
    <*> strOption (long "model-file" <> value "./data/stories15M.bin" <> metavar "MODEL_FILE" <> help "Model binary file")
    <*> option auto (long "temperature" <> value 0.0 <> metavar "TEMPERATURE" <> help "Temperature")
    <*> option auto (long "steps" <> value 256 <> metavar "STEPS" <> help "Number of steps")
    <*> optional (strArgument (metavar "PROMPT" <> help "Initial prompt"))

main :: IO ()
main = do
    Options {..} <- execParser $ info (optionsParser <**> helper) fullDesc
    modelFileHandle <- openFile modelFile ReadMode
    tokenizerFileHandle <- openFile tokenizerFile ReadMode
    tokenizerAbsolutePath <- canonicalizePath tokenizerFile
    printf "loading model file %s\n" modelFile
    modelFileContent <- BS.hGetContents modelFileHandle
    printf "loading tokenizer %s\n" tokenizerAbsolutePath
    tokenizerFileContent <- BS.hGetContents tokenizerFileHandle
    putStrLn "running inference..."
    Inference.run modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed
    hClose modelFileHandle
    hClose tokenizerFileHandle
