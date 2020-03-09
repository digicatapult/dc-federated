"""
Simple runner to start a example global model server.
"""

from dc_federated.example_dcf_model import ExampleGlobalModel


def run():
    """
    This should be run to start the global server to test the backend + model integration
    in a distributed setting. Once the server has started, run the corresponding local-model
    runner(s) in local_model.py in other devices. Run `python federated_local_model.py -h' for instructions
    on how to do so.
    """
    egm = ExampleGlobalModel()
    print("\n************")
    print("Starting an Example Global Model Server to test backend + model integration\n"
          "in a distributed setting. Once the server has started, run the corresponding\n"
          "local-model runner(s) in other devices. Run `python federated_local_model.py -h' for\n"
          "instructions and use the server parameters provided below.")
    print(f"\nServer starting at \n\tserver-host-ip: {egm.server.server_host_ip} \n\tserver-port: {egm.server.server_port}")
    print("\n************\n")

    egm.start()


if __name__ == '__main__':
    run()
