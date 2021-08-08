
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
from pyglet.window import key
import numpy as np
import argparse


from customPong import CustomPong
from neural_network import SlowNeuralNet, FastNeuralNet


def playNN(nn):  
    a = np.array([0.0, 0.1, 0.0])

    def key_press(k, mod):
        if k == key.UP:
            a[0] = +1.0
        if k == key.DOWN:
            a[2] = +1.0

    def key_release(k, mod):
        if k == key.UP:
            a[0] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CustomPong()
    ob_l, ob_r = env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    totalReward = 0
    for step in range(100000000):
        # if step%3 == 0:
        env.render()
        if step % 10 == 0:
            al = a
            ar = nn.getOutput(ob_r)
        (ob_l, ob_r), (r_l, r_r), done, info = env.step((al, ar))
        # print(f"{ob_l} - {ob_r}")
        totalReward += r_r
        if done:
            break
    observation = env.reset()
    print("Expected Fitness of %4d | Actual Fitness = %2f" % (nn.fitness, totalReward))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", action="store_true", help="Set to play against a pretrained model")
    args = parser.parse_args()

    nn = SlowNeuralNet([1])
    if args.use_pretrained:
        # A model is just a set of numbers! below is the entire data needed to store a pong AI!
        json_data = {"fitness": 0.8603600285061459, "nodeCount": [5, 13, 8, 13, 3], "neurons": [["{\"bias\": 0.48309720848883475, \"weights\": [0.5416863215041516, 0.11985546495742039, -0.12935521476344625, -0.15926439582502194, -0.6470014912955415]}", "{\"bias\": 0.18297441295379513, \"weights\": [0.6914469979359279, -0.4006906526509537, 0.5233864301204221, 0.24750723696451904, 0.05828418855407813]}", "{\"bias\": 0.13777740614422562, \"weights\": [0.176631481972928, -0.7622361578940169, -0.631073088517057, 0.23826861874308713, 0.584408860861499]}", "{\"bias\": 0.937484089642787, \"weights\": [-0.23764347628083593, 0.010492180243407923, 0.11716429563789044, -0.14375534480522112, -0.8069381686787749]}", "{\"bias\": -0.5128631336476837, \"weights\": [0.6408124452188437, -0.5479759987915283, -0.5010209212318317, -0.5593875858588806, 0.30249152533468693]}", "{\"bias\": 0.31962300216800377, \"weights\": [-0.16390957268684714, 0.09708057549830929, -0.39270893611731417, 0.22408300431154604, 0.626707641742799]}", "{\"bias\": 0.414481243466426, \"weights\": [0.16000835775561328, 0.17141411500591475, -0.7121429891848401, 0.9291021601785103, -0.5997126601017739]}", "{\"bias\": -0.6301885453016018, \"weights\": [-0.8923061050246599, 0.13428825525541854, -0.1449080734077799, -0.18088303140430328, -0.08077109335420896]}", "{\"bias\": 0.816968737630053, \"weights\": [-0.39857870779111604, -0.5710294693592701, 0.3540432618050684, 0.27297532420909376, 0.7017010080090034]}", "{\"bias\": -0.40898005950491445, \"weights\": [0.7961874191636369, -0.44001250662815394, 0.5572971767729769, 0.8386352462964022, 0.8866765318521717]}", "{\"bias\": -0.5659403895849231, \"weights\": [0.7847841718222961, 0.5923137575530748, -0.6615491068316441, -0.9168903180246712, 0.865520497023637]}", "{\"bias\": 0.03556974937563151, \"weights\": [-0.26577733955505534, 0.29739200015142586, 0.9644600675926711, -0.35146963074142157, 0.9616196500876708]}", "{\"bias\": -0.13442242135618687, \"weights\": [-0.2262202293584752, -0.6948903197978675, -0.7801604066772339, 0.591587108951843, 0.13489447269062915]}"], ["{\"bias\": 0.5948098780651871, \"weights\": [-0.7028121073106188, 0.1653697770580691, -0.9488981163006145, -0.20290369530144492, -0.18679720054259596, 0.995946964888915, 0.8745781115003117, 0.33207732768345277, -0.7224483935214228, 0.9458213797496497, -0.7744475695238571, 0.41113118532807613, -0.33124378902645524]}", "{\"bias\": -0.07763653704052187, \"weights\": [-0.6400081100136574, -0.051695512989318626, -0.16649311960932578, -0.9705990419912771, 0.3229626534308827, -0.439895224072657, -0.7250029783679586, 0.5560163196169365, -0.8469268114061137, -0.6138827798763902, -0.6485865115361698, 0.5971713081981618, -0.4877038384423411]}", "{\"bias\": 0.1745571112766744, \"weights\": [0.836917293179201, 0.5994911359713395, 0.9615613498687086, -0.47244294143996046, 0.827830954330264, -0.2672345171952901, 0.6525330937988059, 0.21688431561009924, 0.552358355420902, 0.6314283383014523, -0.2795221064020441, 0.9517774304034381, -0.4635680528811754]}", "{\"bias\": 0.7184784172287955, \"weights\": [0.44487328834668993, -0.5083776721917386, -0.33098829577254496, -0.7788337044076779, 0.48139024101260275, -0.9794999212008182, 0.8294377897595417, -0.40965538956051506, 0.22257279510081318, 0.8667779132935691, 0.6166708498913624, 0.15770712913030072, -0.7329061238167649]}", "{\"bias\": -0.013315208817757052, \"weights\": [0.09899364217681605, -0.14593795203982696, 0.4085439588473614, 0.8885668797380315, 0.9845417885323013, 0.7996301676634565, 0.5686538527409151, -0.40863436665570063, 0.3165822565047911, 0.4835100777668979, -0.9701591531468554, -0.14980227742377905, -0.7590633689877646]}", "{\"bias\": 0.8939626950503547, \"weights\": [0.827776007126259, 0.623285067693605, 0.7782466555337264, 0.46362376238246195, 0.4023384081932868, -0.914436262963435, -0.039505855423362046, 0.19664153618275426, -0.4802792105939937, 0.9376676405340896, 0.9990856012234961, -0.6797566787602254, -0.17610776495556757]}", "{\"bias\": -0.5118024655422688, \"weights\": [0.8868352241925759, 0.2727105724037824, 0.9073199247646453, 0.2861025137512039, -0.5859003234933715, 0.43143948445656766, -0.38520438155978876, 0.9486067824105009, -0.340610405173015, -0.9151416162349078, -0.21426885741418378, -0.4944071068653628, 0.44092501941476714]}", "{\"bias\": 0.4583868405349072, \"weights\": [-0.15674361172089202, -0.6176885525857927, 0.9134951866144951, 0.06724120638887499, -0.4419043196690269, 0.49674799868207264, 0.7156790368790278, 0.10455535393831084, 0.3859118019789394, 0.23161924766486863, 0.4314941685727085, -0.9831032728423605, -0.4801795189865701]}"], ["{\"bias\": 0.5585355815981965, \"weights\": [0.08635153145293217, 0.3239355808794153, -0.9556239802586071, -0.7970350245966755, -0.22333163577048176, -0.04523304545718854, 0.6705567698651087, 0.833429223915765]}", "{\"bias\": 0.4045284816394499, \"weights\": [0.299544217210046, 0.49502755933606757, -0.5623584273412545, -0.5884973156828606, 0.5417440912660334, -0.9869267205405223, 0.3867204755513891, 0.35516968045581887]}", "{\"bias\": 0.05253094234930322, \"weights\": [0.07439447265257404, -0.04064279959013706, 0.4723009650306058, -0.36492060892908595, 0.6673878726803284, -0.9771520090366801, 0.7347630743130469, -0.31517486248996884]}", "{\"bias\": 0.7538607087956866, \"weights\": [0.04839939960725004, -0.7855555208988854, -0.026553265557742067, 0.624642352186882, 0.4613604010522312, 0.606828328318898, -0.9137419272932745, -0.16638646141136393]}", "{\"bias\": -0.449783839639714, \"weights\": [0.6156066449203108, -0.11434581526811205, -0.19020656643891654, 0.6832046475400124, 0.9834585478282081, -0.47196717295773016, -0.916084719214628, 0.7071303806101419]}", "{\"bias\": -0.8550847979501972, \"weights\": [-0.7321747692570537, -0.22205481534800997, 0.767181655811743, 0.7102393216083811, 0.3650809473395431, 0.7439792228377422, 0.5633188513321628, -0.5846831980693834]}", "{\"bias\": -0.6679672477480998, \"weights\": [0.38479849176246206, 0.07283351140184724, -0.6384841216003059, 0.5613957287086166, -0.25067421371178367, -0.759920415645172, -0.22327113869783743, -0.3485633684300964]}", "{\"bias\": 0.8162154402162873, \"weights\": [-0.46425145697840264, 0.12406938628744446, -0.6485516134865388, -0.8415236585984007, -0.8437695543805048, -0.26188146219463526, 0.28362352725766327, 0.5248297171686684]}", "{\"bias\": 0.9009825098055804, \"weights\": [0.8229083861836768, -0.5807397369134628, -0.3125268855410748, 0.47080211513511294, 0.03742161051652526, 0.678856341360826, -0.7920675629741305, 0.43431526241381246]}", "{\"bias\": 0.0015084839059005262, \"weights\": [0.4610078816998877, -0.5873482140837423, 0.21786029786878314, 0.6664249802452085, -0.24903377191400988, 0.9777241866394131, -0.7650676458173049, 0.6523016388795935]}", "{\"bias\": -0.24883244110526515, \"weights\": [0.5803018792632912, 0.7505376196093252, 0.5092318180555131, 0.7045258531951024, 0.39225409628608254, -0.5114748824139657, -0.06836249165897224, 0.5747076307233387]}", "{\"bias\": -0.9727763863017216, \"weights\": [-0.3251605287132058, -0.5339775822155544, 0.05327684770566665, -0.17287115737368564, 0.36305357379039815, -0.2173510306152573, 0.294093377081174, -0.469457590593513]}", "{\"bias\": 0.6321291828384115, \"weights\": [0.7459212619890161, -0.7740875667755458, -0.8944454734554719, -0.33513303011662066, 0.05655875018030643, 0.5853228950498375, 0.630651091920938, -0.6218285160634509]}"], ["{\"bias\": 9.75199188790743e-05, \"weights\": [-0.9598764533949564, 0.607370724455859, -0.6472015934018969, -0.6246715985210183, 0.4410403039394477, -0.9289812848349519, 0.5243698449035241, -0.243995182702889, -0.5209103208478592, 0.9040214101660022, 0.38666765717844576, -0.21151579426590783, -0.9130206041274451]}", "{\"bias\": 0.7650743388237087, \"weights\": [0.8863414453344165, 0.5155795225476336, 0.9344848522948193, -0.7204177092479518, 0.6306818399295433, 0.4414972882137649, 0.2857311311367119, -0.5202139814603886, -0.8416846452615516, 0.5584717939456232, -0.06310461276234292, 0.16578605984561912, -0.1437366506592146]}", "{\"bias\": 0.8528570128298738, \"weights\": [0.8179845219624908, 0.7620596356204328, 0.776869612217902, -0.008538105575272459, 0.5092516971509369, -0.7446092809757816, -0.03560661924591435, 0.23659962450574534, 0.3692492742683131, 0.28905237874891476, 0.07660005157042193, -0.7834224020004938, -0.9916785144804796]}"]], "seed": 8812}
        nn.from_json_data(json_data)
    else:
        nn.from_json("saved_models/learned_pong_nn.json")

    playNN(nn)