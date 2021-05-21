import math


def main(papers_values):
    max_mse = max(list(map(lambda x: x["MSE"], papers_values)))
    min_mse = min(list(map(lambda x: x["MSE"], papers_values)))

    def turn_mse_to_percent(paper):
        paper["MSE_PERCENT"] = ((paper["MSE"] - min_mse)/(max_mse - min_mse))*1
        return paper

    def square_dist(paper):
        paper["result"] = paper["SSIM"] - paper["MSE_PERCENT"]
        paper["feature"] = paper["feature"]
        return paper
    papers = list(map(turn_mse_to_percent, papers_values))
    papers = list(map(square_dist, papers))
    # print('result:' + str(papers))
    return max(papers, key=lambda x: x["result"])
