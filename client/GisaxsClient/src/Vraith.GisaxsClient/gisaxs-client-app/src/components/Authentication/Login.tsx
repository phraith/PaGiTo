
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import TextField from "@mui/material/TextField"
import Stack from "@mui/material/Stack"
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
    fetch(url, requestOptions)
      .then((response) => {
        if (!response.ok) { throw new Error(response.status.toString()) }
        return response.json();
      })
      .then((data) => {
        localStorage.setItem('apiToken', data.token)
        window.dispatchEvent(new Event("storage"));
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
