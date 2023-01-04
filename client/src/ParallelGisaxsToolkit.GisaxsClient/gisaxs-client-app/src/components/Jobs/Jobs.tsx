import MiniDrawer from "../Drawer/MiniDrawer";
import CssBaseline from "@mui/material/CssBaseline"

import { useEffect, useState } from "react";
import * as React from "react";
import Grid from "@mui/material/Grid/Grid";
import JobsTable from "./JobsTable";
import { JobInfo } from "../../utility/JobInfo";
import { JsonViewer } from '@textea/json-viewer'
import Box from "@mui/material/Box/Box";
import Button from "@mui/material/Button/Button";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";

const Jobs = () => {
    const [jobInfo, setJobInfo] = React.useState<JobInfo>(new JobInfo(0, { body: "{}" }))

    const receiveJobResult = (message: any) => {
        console.log(message)
    };

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            receiveJobResult,
            (message: string) => { },
            (message: string) => { }
        )
    )

    useEffect(() => {
        hubConnection.connect()
      }, [hubConnection]);

    const sendJob = () => {
        let jsonConfig = JSON.stringify({
            info: {
                clientId: 0,
                jobId: 0,
                jobType: "fit"
            },
            config: JSON.parse(jobInfo.info.body)

        });
        hubConnection.requestJob(jsonConfig, "");
    };

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={12} sm={12} md={12} lg={8}>
                    <Box display="flex" sx={{ flexDirection: "column", padding: 10 }}>
                        <Box display="flex" sx={{ paddingBottom: 1, width: "100%" }}>
                            <JobsTable setJobsInfo={(updatedJobInfo: JobInfo) => { console.log(updatedJobInfo); setJobInfo(updatedJobInfo) }} />
                            <Box sx={{ height: 500, overflow: 'auto', paddingLeft: 5, width: "50%" }}>
                                <JsonViewer value={JSON.parse(jobInfo.info.body)} />
                            </Box>
                        </Box>
                        <Button onClick={sendJob}>
                            Send selected job
                        </Button>
                    </Box>
                </Grid>
            </Grid>
        </React.Fragment >
    );
};

export default Jobs