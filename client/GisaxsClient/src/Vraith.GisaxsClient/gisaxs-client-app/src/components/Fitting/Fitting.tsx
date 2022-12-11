import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import throttle from "lodash/throttle";
import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, { useEffect, useRef, useState } from "react";
import Sample from "../Sample/Sample";
import LineProfileWrapper from "../ScatterImage/LineProfileWrapper";
import { Coordinate, LineMode, LineProfile, LineProfileState } from "../../utility/LineProfile";
import ImageTable from "./ImageTable";
import { Button } from "@mui/material";
import LineProfileGraphVx from "./LineProfileGraphVx";
import { ImageInfo } from "../../utility/ImageInfo";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import ColormapSelect from "../Colormap";


const Fitting = () => {
    const receiveJobInfos = (message: any) => {
        setCurrentInfoPath(message)
        setIsActive(true)
    }

    const getLineprofiles = (hash: any) => {
        let url = `/api/redis/data?hash=${hash}`;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                let json = JSON.parse(data);




                let traces = []
                let values = json.modifiedData
                let k = values.map((x: number, index: number) => { return { x: index, y: x } })
                traces.push(k)
                setSimulatedPlotData(traces[0])
            })
    }

    const receiveJobResult = (hash: any, colormap: any) => {
        let url = `/api/redis/image?colorMapName=${colormap}&hash=${hash}`;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => handleData(data));
    };

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            receiveJobResult,
            receiveJobInfos,
            getLineprofiles
        )
    )

    const [intensities, setIntensities] = useState<string>();
    const [refIntensities, setRefIntensities] = useState<string>();
    const [currentInfoPath, setCurrentInfoPath] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(LineMode.Horizontal, [], new LineProfile(new Coordinate(0, 0), new Coordinate(0, 1))));
    const [simulatedPlotData, setSimulatedPlotData] = React.useState([])
    const [realPlotData, setRealPlotData] = React.useState([])

    const [openTable, setOpenTable] = React.useState<boolean>(false)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(new ImageInfo(0, 0, 0))

    useEffect(() => {
        hubConnection.connect()
    }, [hubConnection]);

    const handleData = (input: any) => {
        let startTime = performance.now();
        let json = JSON.parse(input);
        setIntensities(json.data);
        setImgWidth(json.width);
        setImgHeight(json.height);
        let endTime = performance.now();
        console.log(`Handling data took ${endTime - startTime} milliseconds`);
    };

    const [colormap, setColorMap] = React.useState("twilightShifted");
    const [jsonData, setJsonData] = React.useState({});
    const [isActive, setIsActive] = React.useState(false);

    const jsonCallback = (value, key) => {
        jsonData[key] = value;
        setJsonData({ ...jsonData });
    };

    const sendLineprofileRequest = (json) => {
        hubConnection.requestProfiles(json);
    }

    const throttled = useRef(throttle((jsonConfigForSimulation, jsonConfigForRealImage) => {
        sendLineprofileRequest(jsonConfigForSimulation)

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfigForRealImage
        };
        let url = `/api/scatterstore/profile`;
        fetch(url, requestOptions)
            .then((response) => response.json())
            .then((data) => {
                let json = JSON.parse(data);
                let traces = []
                let values = json.modifiedData
                let k = values.map((x: number, index: number) => { return { x: index, y: x } })
                traces.push(k)
                setRealPlotData(traces[0])
            })


    }, 250));

    useEffect(() => {
        let jsonConfigForSimulation = JSON.stringify({
            clientInfo: {
                jobType: "simulation"
            },
            jobInfo: {
                clientId: 0,
                jobId: 0,
                intensityFormat: "doublePrecision",
                simulationTargets: [
                    lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
                ]
            },
            config: {
                ...jsonData,
            },
        });

        let jsonConfigForRealImage = JSON.stringify({
            id: imageInfo.id,
            target: lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
        })

        throttled.current(jsonConfigForSimulation, jsonConfigForRealImage)
    }, [lineprofileState?.currentLineProfile, jsonData]);

    useEffect(() => {
        jsonCallback(lineprofileState.lineProfiles.map(lp => {
            return {
                start: lp.start,
                end: lp.end
            }
        }), "lineprofiles")

        jsonCallback(imageInfo.id, "imageId")
    }, [lineprofileState.lineProfiles])

    useEffect(() => {
        let jsonConfig = JSON.stringify({
            clientInfo: {
                jobType: "simulation"
            },
            jobInfo: {
                clientId: 0,
                jobId: 0,
                intensityFormat: "greyscale",
                simulationTargets: [
                    // { start: { x: 0, y: 0 }, end: { x: 1475, y: 120 } }
                ]
            },
            config: {
                ...jsonData,
            },
        });
        localStorage.setItem("simulation_config", jsonConfig);
        hubConnection.requestJob(jsonConfig, colormap);

    }, [jsonData, colormap]);

    useEffect(() => {
        let url = "/api/scatterstore/get?id=" + imageInfo.id + "&colormap=" + colormap;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => setRefIntensities(data));
    }, [imageInfo.id, colormap]);

    const sendJobInfo = () => {
        let jsonConfig = JSON.stringify({
            info: {
                body: JSON.stringify(jsonData),
                history: []
            },
            userId: 0
        });

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfig
        };
        let url = "/api/jobstore/push";
        fetch(url, requestOptions)
            .then(data => console.log(data));
    }

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={6} sm={6} md={6} lg={8}>
                    <Box display="flex" sx={{ gap: 2, padding: 10 }}>
                        <LineProfileWrapper width={imgWidth} height={imgHeight} profileState={lineprofileState}
                            setProfileState={setLineprofileState}>
                            <ScatterImage intensities={intensities} width={imgWidth} height={imgHeight} />
                        </LineProfileWrapper>

                        <LineProfileWrapper width={imgWidth} height={imgHeight} profileState={lineprofileState}
                            setProfileState={setLineprofileState}>
                            <ScatterImage intensities={refIntensities} width={imgWidth} height={imgHeight} />
                        </LineProfileWrapper>
                    </Box>
                </Grid>
                <Grid item xs={12} sm={12} md={12} lg={4}>
                    <Box display="flex" sx={{ flexDirection: "column", gap: 2, padding: 10 }}>
                        <Box display="flex" sx={{ paddingBottom: 1 }}>
                            {!openTable &&
                                <LineProfileGraphVx simulatedData={simulatedPlotData} realData={realPlotData}  ></LineProfileGraphVx>
                            }
                            {openTable &&
                                <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { console.log(updatedImageInfo); setImageInfo(updatedImageInfo) }} />
                            }
                        </Box>
                        <Box display="flex" sx={{ paddingBottom: 1, gap: 2 }}>
                            <Instrumentation jsonCallback={jsonCallback} initialResX={imageInfo.width} initialResY={imageInfo.height} />
                            <UnitcellMeta jsonCallback={jsonCallback} />
                        </Box>
                        <Box display="flex" sx={{ paddingBottom: 1 }}>
                            <ColormapSelect colormap={colormap} setColormap={setColorMap} />
                            <Button onClick={() => sendJobInfo()}>
                                Create Job Description
                            </Button>
                            <Button onClick={() => { setOpenTable(prevState => !prevState) }}>
                                {openTable ? "Show graph" : "Show images"}
                            </Button>
                        </Box>
                        <Grid container spacing={2}>
                            <Grid item xs={7} sm={7} md={7} lg={7}>
                                <GisaxsShapes isSimulation={false} jsonCallback={jsonCallback} />
                            </Grid>
                            <Grid item xs={5} sm={5} md={5} lg={5}>
                                <Sample jsonCallback={jsonCallback} />
                            </Grid>
                        </Grid>
                    </Box>
                </Grid>
            </Grid>
        </React.Fragment>
    );
};

export default Fitting