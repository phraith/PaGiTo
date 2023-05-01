import React from "react";

// interface JsonCallback {
//     (value: any, key: any): void;
//   }

const useJobEffect = (config: {}, colormap: string): any => {
    const [response, setResponse] = React.useState({});

    React.useEffect(() => {
        let jsonConfig = JSON.stringify(
            {
                jsonConfig: JSON.stringify({
                    meta: {
                        type: "simulation",
                        notification: "receiveJobResult",
                        colormap: colormap
                    },
                    properties: {
                        intensityFormat: "greyscale",
                        simulationTargets: []
                    },
                    config: {
                        ...config,
                    },
                })
            });

        localStorage.setItem("simulation_config", jsonConfig);

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfig
        };

        const url = "/api/job";
        fetch(url, requestOptions)
            .then(response => setResponse(response));
    }, [config, colormap]);

    return response;
}

export default useJobEffect