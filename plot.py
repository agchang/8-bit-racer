import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plotComparison():
    sort = True 
    N = 25
    ind = np.arange(N)
    scoresB = [float(s) for s in open('scores/scores-random.txt', 'r').read().splitlines()]
    scoresQ = [float(s) for s in open('scores/scores-best.txt', 'r').read().splitlines()]
    scoresO = [float(s) for s in open('scores/scores-oracle.txt', 'r').read().splitlines()]

    scoresB = scoresB[0:N]
    scoresQ = scoresQ[0:N]
    scoresO = scoresO[0:N]

    if sort:
        scoresB = sorted(scoresB, reverse=True)
        scoresQ = sorted(scoresQ, reverse=True)
        scoresO = sorted(scoresO, reverse=True)

    ax = plt.subplot(111)
    w = 0.3
    oracle = ax.bar(ind-w, scoresO, width = 0.3, color='green')
    q = ax.bar(ind, scoresQ, width = 0.3, color='red')
    baseline = ax.bar(ind+w, scoresB, width = 0.3, color='blue')

    ax.legend((oracle[0], q[0], baseline[0]), ('Oracle', 'DQN', 'Baseline'))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Baseline/Oracle Comparison')
    plt.show()

def plotScoreAverages(scores, averageInterval=50):
    averages = []
    numScores = len(scores)
    for i in range(0, numScores, averageInterval):
        batch = scores[i:i+averageInterval] 
        average = sum(batch)/float(len(batch))
        averages.append(average)
    plt.xlabel('Batches of 50 episodes')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.plot(range(len(averages)), averages)
    plt.show()

if __name__ == "__main__":
    scores = [float(s) for s in open('scores.txt', 'r').read().splitlines()]
    plotScoreAverages(scores, averageInterval=50)
    plotComparison()
