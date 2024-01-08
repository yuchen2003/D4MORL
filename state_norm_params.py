import numpy as np

state_norm_params = {
    "MO-Hopper-v2": {
        "mean": np.array([1.6910676670564566, 0.7188205598476783, 0.5322217970635693, -0.12927024149323435, -0.2669105396279655, 3.470458761569363, 0.06018225886384595, 0.21065001052872792, 0.12166325365667992, 0.0022834114901502368, 0.533496662012031, ]),
        "var": np.array([0.2317919090993381, 0.11364615084946238, 0.08642215993352258, 0.15678759466172934, 0.40362635956893533, 2.0287736251224544, 6.675020099891263, 8.074847876766237, 4.851578526302009, 9.241463849187086, 24.126371929882126, ])
    },
    "MO-Hopper-v3": {
        "mean": np.array([1.5240508784185178, 0.13842510326685742, -0.17786304221128424, 0.038771794675877505, -0.10343764268102175, 2.4920178804707636, 0.027017394434642326, 0.04743306099090415, -0.042258853691565706, 0.002226240393204824, 0.3339253214321786, ]),
        "var": np.array([0.15873789173388206, 0.07530984911532675, 0.04964430130216111, 0.06624363843649671, 0.36971900290448684, 1.580305049446974, 4.724374393138628, 4.2410311273835335, 2.596440031101406, 3.6448187817761575, 17.530621559474454, ])
    },
    "MO-Ant-v2": {
        "mean": np.array([0.5542124963409156, 0.7999683555762525, 0.009863096485096238, -0.007305259670295653, -0.22241129449351543, -0.14625630144678578, 0.7285598258612382, -0.15622854090843116, -0.7558671210589664, -0.18992007232190256, -0.636804406513384, 0.1114969947980098, 0.695150499706359, 1.828306087499514, 2.159629689400505, -0.00840197083355858, 0.0003837592849895247, -0.0004847142610150224, -0.017385182697877703, 0.0031617776037215824, 0.026850642052232115, -0.007624756189907075, -0.025207795753963855, -0.0044391771108651685, -0.026176446059146985, 0.004550679291712785, 0.023817902448263924]),
        "var": np.array([0.019909349621709146, 0.10832151999982116, 0.056293553091275185, 0.05008568243614873, 0.07060612924859021, 0.13408908334395445, 0.05797934328619498, 0.13208083663384665, 0.06203176714065863, 0.13711340787979234, 0.04571036247570691, 0.14164111484168748, 0.056869464282972966, 2.3703829306702646, 2.225608956137697, 1.0048458124905235, 1.4201285340139262, 1.1018409379914178, 0.8371364916038128, 12.88260277108399, 6.050218642320978, 13.697993792377657, 7.628382667541564, 14.793005962058137, 2.79973288056342, 13.757827405716325, 3.856238518447082])
    },
    "MO-HalfCheetah-v2": {
        "mean": np.array([-0.11039155649562715, 0.042052043095244546, 0.007974530080267026, 0.025969403818104805, -0.024816826751808352, -0.0636500514191894, -0.10298306479632521, -0.09636814694335218, 1.1674846767784985, -0.01600160475121651, 0.0025169242774512623, 0.014293437516297478, -0.04704402668655135, 0.023806862084637732, -0.018959193549851296, -0.01055708606529066, -0.009011287377515538, ]),
        "var": np.array([0.004391946018861979, 0.014677443105019028, 0.047345551039276484, 0.034254120250921095, 0.04647823856515354, 0.08476213436456903, 0.062186602246775115, 0.04672259487223586, 1.2245661950907083, 0.4502638757249774, 2.43275886405476, 17.235111796996094, 20.711672487474573, 25.233969219954208, 28.238632724203335, 25.138293522035838, 19.430287464975926, ])
    },
    "MO-Swimmer-v2": {
        "mean": np.array([0.20462862976477872, -0.17655695609072317, -0.0914579731566402, 0.28517947356725826, -0.01496406701953703, 0.010407850694912056, -0.008255642744355097, -0.00458389681254946, ]),
        "var": np.array([1.6747366420128713, 1.046295319146989, 0.7075581332532797, 0.531676491286163, 2.809670583640691, 4.164214381986462, 7.1681710050044, 10.684866014656158, ])
    },
    "MO-Walker2d-v2": {
        "mean": np.array([1.116750731036977, -0.3363452709970974, -0.3316850051534519, -0.7511427047318645, 0.19274933189219556, -0.6200324701020578, -0.16067687030963107, 0.5416847411428887, 1.7497760667774616, -0.06936457985904049, -0.11816332219093974, -0.15284182801522847, -0.2933337303981446, -0.05545796533726404, -0.2226723410943518, -0.08635534038498599, 0.1697440381383936, ]),
        "var": np.array([0.0034804486813955443, 0.04143357420205374, 0.08790632324221213, 0.3284840162691254, 0.3090904753830096, 0.09216643705539583, 0.04303564000987159, 0.10557799161138029, 1.3887131180405956, 0.2768508061213584, 7.215076808820878, 13.892186424782372, 26.37067211386862, 30.099564927710077, 10.967798122337761, 5.309070246017233, 22.205671104177277, ])
    }
}