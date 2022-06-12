
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import Collapse from "@mui/material/Collapse"
import FormControl from "@mui/material/FormControl"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import TextField from "@mui/material/TextField"
import Stack from "@mui/material/Stack"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import React from "react";

const Login = () => {
  const [currentPassword, setCurrentPassword] = React.useState("");
  const [currentUsername, setCurrentUsername] = React.useState("");

  const handleLogin = () => {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: currentUsername,
        password: currentPassword,
      }),
    };

    let url = "/api/auth/login";
    fetch(url, requestOptions).then((response) => response.text()).then((token) => {
        localStorage.setItem('apiToken', token)
    })
  };

  return (
    <Card>
      <Stack>
        <ListItem>
          <TextField
            id="outlined-username-input"
            label="Username"
            onChange={(e) => setCurrentUsername(e.target.value)}
          >
            User
          </TextField>
        </ListItem>
        <ListItem>
          <TextField
            type="password"
            id="outlined-password-input"
            label="Password"
            onChange={(e) => setCurrentPassword(e.target.value)}
          >
            Password
          </TextField>
        </ListItem>
        <ListItem>
          <Button onClick={handleLogin}>Login</Button>
        </ListItem>
      </Stack>
    </Card>
  );
};

export default Login;
