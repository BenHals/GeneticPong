# GeneticPong


## Installation Instructions (WINDOWS)

1. Install Python (3.5+)
2. Install Conda
3. Install Microsoft Visual C++ Build Tools for Visual Studio 2017 ([HERE](https://visualstudio.microsoft.com/downloads/) - Scroll down page a bit)
4. Create new environment: `conda create -n DEVSAI python=3.7 pip`
5. Activate environment: `conda activate DEVSAI`
6. Install Python Reqs
- `pip install gym`
- `pip install pystan`
<!-- 9) `pip install swig` -->
- `pip install Box2D`
- `pip install gym[all]`
- `pip install pyglet==1.2.4`
- `pip install gym[box2d]`
- `pip install tqdm`


## To run

1. Try playing Pong against a human (or yourself)
    - run `python play_pong_human.py`
    - Use <W, S> keys to move left paddle Up and Down, use Up Arrow and Down Arrow to move right paddle

2. Try playing against a pretrained AI
    - run `python play_pong_AI.py --use_pretrained`
    - Use arrow keys to move your paddle, as the left player

3. Try training your own AI
    - run `python learn_pong.py`
    - Training will start - each generation will show a progress bar
    - When a generation ends, the best member will be saved in `saved_models/learned_pong_nn.json`
    - You can play against this model by running `python play_pong_AI.py` in another window (remember to have the other window in this repository, and have activated the conda environment)
    - When training completes, you can press any key to show the two best agents competing
    - Adjust the `MAX_GENERATIONS` parameter to a lower value to complete at your desired generation

4. Try filling in the code yourself! You can find the code skeleton files in the `Make It Yourself` folder
    - TODO

5. This process is general! We can evolve a NN to walk as well, using the same model and code!
    - Run `learn_walker.py`
    - TODO