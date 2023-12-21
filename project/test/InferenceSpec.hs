module InferenceSpec (spec) where

import Test.Hspec
import Inference (replaceAtIndex)

spec :: Spec
spec = do
  describe "replaceAtIndex" $ do
    it "replaces a value" $ do
      replaceAtIndex 1 3.0 [1.0, 2.0, 3.0] `shouldBe` [1.0, 3.0, 3.0]

    it "works at the end of the list" $ do
      replaceAtIndex 2 0.5 [1.0, 2.0, 3.0] `shouldBe` [1.0, 2.0, 0.5]
