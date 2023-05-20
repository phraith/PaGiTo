import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import _debounce from "lodash/debounce"
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import { useEffect, useRef, useState } from "react";
import * as React from "react";
import Sample from "../Sample/Sample";
import { LineMode, LineProfileState } from "../../utility/LineProfile";
import ImageTable from "./ImageTable";
import { Button } from "@mui/material";
import { ImageInfo } from "../../utility/ImageInfo";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import ColormapSelect from "../Colormap";
import ScatterImageWithLineprofile from "../ScatterImage/ScatterImageWithLineprofile";
import LineProfileGraph from "./LineProfileGraphECharts";
import useJsonCallback from "../../hooks/useJsonCallback";
import useJobEffect from "../../hooks/useJobEffect";
import Settings from "../Settings";

const Fitting = () => {
    const getLineprofiles = (hash: any) => {
        const requestOptions1 = {
            method: 'POST',
            headers:
            {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    jobId: hash,
                    includeResult: true
                }
            )
        };

        let url1 = "/api/job/state";
        fetch(url1, requestOptions1)
            .then((response) => response.json())
            .then((data) => {
                let json = JSON.parse(data.job.result)
                let traces = []
                let values = json.numericResults[0].modifiedData
                let k = values.map((x: number, index: number) => { return [index, x] })
                traces.push(k)
                setSimulatedPlotData(traces[0])
            })
    }

    const receiveJobResult = (hash: any) => {
        const requestOptions1 = {
            method: 'POST',
            headers:
            {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    jobId: hash,
                    includeResult: true
                }
            )
        };

        let url1 = "/api/job/state";
        fetch(url1, requestOptions1)
            .then((data) => data.json())
            .then((data) => handleData(data));
    };

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            [
                ["receiveJobResult", (message: string) => receiveJobResult(message)],
                ["receiveLineProfile", (message: string) => getLineprofiles(message)]
            ]
        )
    )

    const [intensities, setIntensities] = useState<string>();
    const [refIntensities, setRefIntensities] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(LineMode.Horizontal, [], null));
    const [simulatedPlotData, setSimulatedPlotData] = React.useState([])
    const [realPlotData, setRealPlotData] = React.useState([])
    const [openTable, setOpenTable] = React.useState<boolean>(true)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(null)
    const [json, jsonCallback] = useJsonCallback();
    const [colormap, setColorMap] = React.useState("twilightShifted");
    const reponse = useJobEffect(json, colormap);

    useEffect(() => {
        hubConnection.connect()
    }, [hubConnection]);

    const handleData = (input: any) => {
        let startTime = performance.now();
        let json = JSON.parse(input.job.result)
        setIntensities(json.jpegResults[0].data);
        setImgWidth(json.jpegResults[0].width);
        setImgHeight(json.jpegResults[0].height);
        let endTime = performance.now();
        console.log(`Handling data took ${endTime - startTime} milliseconds`);
    };



    const requestLineProfiles = (jsonConfigForSimulation, jsonConfigForRealImage) => {
        const requestOptions1 = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    "jsonConfig": jsonConfigForSimulation
                }
            )
        };

        let url1 = "/api/job";
        fetch(url1, requestOptions1)

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfigForRealImage
        };
        const url = `/api/image/profile`;
        fetch(url, requestOptions)
            .then((response) => response.json())
            .then((data) => {
                let traces = []
                let values = data.modifiedData
                let k = values.map((x: number, index: number) => { return [index, x] })
                traces.push(k)
                setRealPlotData(traces[0])
            })
    }


    const debounced = useRef(_debounce((jsonConfigForSimulation, jsonConfigForRealImage) => {
        requestLineProfiles(jsonConfigForSimulation, jsonConfigForRealImage)
    }, 50))

    useEffect(() => {
        if (lineprofileState.currentLineProfile === null || imageInfo === null) {
            return;
        }


        console.log(lineprofileState?.currentLineProfile)
        let jsonConfigForSimulation = JSON.stringify({
            meta: {
                type: "simulation",
                notification: "receiveLineProfile"
            },
            properties: {
                intensityFormat: "doublePrecision",
                simulationTargets: [
                    lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
                ]
            },
            config: {
                ...json,
            },
        });

        let jsonConfigForRealImage = JSON.stringify(
            {
                target: {
                    id: imageInfo.id,
                    target: lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
                }
            })

        debounced.current(jsonConfigForSimulation, jsonConfigForRealImage)
    }, [lineprofileState?.currentLineProfile, json]);

    useEffect(() => {
        jsonCallback(lineprofileState.lineProfiles.map(lp => {
            return {
                start: lp.start,
                end: lp.end
            }
        }), "lineprofiles")
    }, [lineprofileState.lineProfiles])

    useEffect(() => {
        if (imageInfo == null) {
            return;
        }


        const url = `/api/image/${imageInfo.id}/${colormap}`;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                setRefIntensities(data.imageAsBase64)
                setImgWidth(data.width);
                setImgHeight(data.height);
            }
            );
    }, [imageInfo, colormap]);

    const sendJobInfo = () => {
        let jsonConfig = JSON.stringify({
            meta: {
                type: "fitting",
                notification: "receiveJobResult",
                persist: true,
                execute: false,
                colormap: colormap
            },
            properties: {
                intensityFormat: "doublePrecision",
                imageId: imageInfo.id,
                simulationTargets: lineprofileState.lineProfiles
            },
            config: {
                ...json,
            },
        });

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    "jsonConfig": jsonConfig
                }
            )
        };
        const url = "/api/job";
        fetch(url, requestOptions)
            .then(data => console.log(data));
    }

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Box display={"flex"} flexDirection={"row"} padding={10} gap={2} sx={{ height: "100vh" }}>
                <Box display={"flex"} flexDirection={"row"} gap={2} sx={{ height: "100%", width: "70%" }}>
                    <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={refIntensities} />
                    <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={intensities} />
                </Box>

                <Box display={"flex"} flexDirection={"column"} gap={2} sx={{ height: "100%", width: "30%" }}>
                    <Box display={"flex"} flexDirection={"row"} gap={2} sx={{ height: "35%" }}>
                        {!openTable
                            ? <LineProfileGraph simulatedData={simulatedPlotData} realData={realPlotData} />
                            : <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { setImageInfo(updatedImageInfo) }} />
                        }
                    </Box>

                    <Box display={"flex"} flexDirection={"row"} sx={{ height: "5%" }} justifyContent={"space-between"}>
                        <Button variant="contained" onClick={() => sendJobInfo()}>
                            Create
                        </Button>
                        <Button variant="contained" onClick={() => { setOpenTable(prevState => !prevState) }}>
                            {openTable ? "Graph" : "Images"}
                        </Button>
                    </Box>
                    <Box sx={{ height: "60%" }}>
                        <Settings isSimulation={false} jsonCallback={jsonCallback} colormap={colormap} setColorMap={setColorMap} />
                    </Box>
                </Box>
            </Box >
        </React.Fragment >
    );
};

export default Fitting