import argparse
import os

parser = argparse.ArgumentParser()

demos = ["main", "intro", "voxelmorph", "inverse", "experiment", "mnist_vxm", "mnist_inv"]
parser.add_argument("-d", "--demo", metavar="NAME", choices=demos, default="main",
                    help="choose a demo to show, among: " + ", ".join(demos))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()
    os.system(f"streamlit run scripts/demo/demo__{args.demo}.py")
