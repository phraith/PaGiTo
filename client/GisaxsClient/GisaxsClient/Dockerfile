#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["GisaxsClient/GisaxsClient.csproj", "GisaxsClient/"]
RUN dotnet restore "GisaxsClient/GisaxsClient.csproj"
COPY . .
WORKDIR "/src/GisaxsClient"
RUN dotnet build "GisaxsClient.csproj" -c Release -o /app/build

FROM build AS publish
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash
RUN apt-get update 
RUN apt-get install -y nodejs

RUN dotnet publish "GisaxsClient.csproj" -c Release -o /app/publish


FROM base AS final
# RUN apt-get update && apt-get install -y redis-tools
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "GisaxsClient.dll"]