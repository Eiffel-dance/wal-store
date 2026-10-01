import unittest
import app
class SmokeTest(unittest.TestCase):
    def test_import(self): self.assertTrue(app)
if __name__=='__main__': unittest.main()
