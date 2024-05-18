import matplotlib.pyplot as plt

x = [i for i in range(25)]
l1 = [0.9103235630510174,
    0.8753292045780375, 0.8735327404758116,
    0.8736913746544893, 0.836417309189252,
    0.8301059367450451, 0.8396045263361668,
    0.8470054826518737, 0.8644262769997967,
    0.8690466034458488, 0.8538733741285636,
    0.9016172072974926, 0.8969652039880958,
    0.9083434190906845, 0.8883548884156139,
    0.9003078606613861, 0.8766049060463821,
    0.8703332555245978, 0.8698224579735603,
    0.8744421131517907,  0.863770033869742,
    0.8960182430026592, 0.885997930467385,
    0.8697932125296275, 0.8688372685756118] #it-it
l2 = [0.8125584757098204,
    0.7446743996302801, 0.8027620088302478,
    0.7502984174413896, 0.720195952858968,
    0.6416933714223548, 0.5829894542976768,
    0.6375157986386875, 0.5952618553301422,
    0.6707484055707406, 0.7254971694312712,
    0.8273555044459819, 0.9055569946626709,
    0.9133125981182305, 0.8892286994415002,
    0.8565432483861255, 0.8627753619056581,
    0.8314414504382694, 0.85433344853502,
    0.8486839702158624, 0.7803692400569028,
    0.8382605187792178, 0.8184113448729408,
    0.8272586875790348, 0.6450225559571647
]#it-en
l3 = [0.8581421526603004,
      0.7416683405350156, 0.6732422823582669,
      0.6685068187385037, 0.5965287572839233,
      0.41961790593098014, 0.4741853615504194,
      0.629378098458223, 0.506515615996682,
      0.4312506873355417,  0.7016398633846125,
      0.9288291863953028, 0.9414219973000548,
      0.9319094750676304, 0.9314184788880461,
      0.914304067908404, 0.9370509861757865,
      0.8960229390681003, 0.9019179599857232,
      0.9002218082515929, 0.8732612138570772,
      0.8455996339249306, 0.7607242795469227,
      0.832667346086913, 0.6307785044293901
]#it-sv

plt.plot(x,l1,'s-',color = 'r',label="it-it")
plt.plot(x,l2,'o-',color = 'g',label="it-en")
plt.plot(x,l3,'^-',color = 'b',label="it-sv")
plt.xlabel("layer")
plt.ylabel("f1")
plt.legend(loc='best')
plt.xticks(range(0,25))
plt.title('trained on Italian')
plt.show()