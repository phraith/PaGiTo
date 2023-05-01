import MiniDrawer from "../Drawer/MiniDrawer";
import CssBaseline from "@mui/material/CssBaseline"
import { useEffect, useState } from "react";
import * as React from "react";
import Grid from "@mui/material/Grid/Grid";
import JobsTable from "./JobsTable";
import { JsonViewer } from '@textea/json-viewer'
import Box from "@mui/material/Box/Box";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import Card from "@mui/material/Card/Card";

const Jobs = () => {
    const [jsonData, setJsonData] = React.useState<string>("{}")
    const [resultNotifier, setResultNotifier] = React.useState({});
    const notifyResult = React.useCallback(() => setResultNotifier({}), []);

    const receiveJobResult = (message: any) => {
        notifyResult()
    };

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            [
                ["receiveJobResult", (message: string) => receiveJobResult(message)],
            ]
        )
    )

    useEffect(() => {
        hubConnection.connect()
    }, [hubConnection]);

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={12} sm={12} md={12} lg={8}>
                    <Box display="flex" sx={{ flexDirection: "column", padding: 10 }}>
                        <Box display="flex" sx={{ paddingBottom: 1, gap: 10 }}>
                            <Card sx={{ width: "100%" }}>
                                <JobsTable resultNotifier={resultNotifier} setJsonData={(data: any) => { setJsonData(data) }} />
                            </Card>
                            <Card sx={{ width: "100%", height: 500, overflow: 'auto', paddingLeft: 5 }}>
                                <JsonViewer value={JSON.parse(jsonData)} />
                            </Card>
                        </Box>
                    </Box>
                </Grid>
            </Grid>
        </React.Fragment >
    );
};

export default Jobs