import torch
from factor_graph import FactorGraph


class TrackBackend:
    def __init__(self, net, video, config):
        self.video = video
        self.update_op = net.update

        self.backend_thresh = config["backend_thresh"]
        self.backend_radius = config["backend_radius"]
        self.backend_nms = config["backend_nms"]

    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """
        torch.cuda.empty_cache()

        t = self.video.counter.value
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=min(1e4, 20*t))
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, backend=True)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
