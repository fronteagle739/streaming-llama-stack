// Local projected image spans, created by the original ds4 vision encoder.
// Files remain under the parent's read leases throughout the producer lifetime.
inline std::vector<llama_dsv41_image_span> studio_vision_spans(const std::string&descriptor,uint32_t context){
    const auto path=std::filesystem::u8path(descriptor);
    if(std::filesystem::file_size(path)>65536)throw std::runtime_error("Vision descriptor exceeds64KiB");
    std::ifstream input(path);auto rows=nlohmann::json::parse(input);
    if(!rows.is_array()||rows.empty()||rows.size()>16)throw std::runtime_error("Vision requires1..16 image spans");
    std::vector<llama_dsv41_image_span>result;uint64_t previous=0;
    for(const auto&row:rows){
        if(!row.is_object()||row.size()!=4||!row.contains("position")||!row.contains("count")||!row.contains("file")||!row.contains("sha256"))
            throw std::runtime_error("Malformed vision span descriptor");
        const auto position=row.at("position").get<uint64_t>(),count=row.at("count").get<uint64_t>();
        if(!count||count>32768||position<previous||position>=context||count>context-position)throw std::runtime_error("Overlapping or oversized image span");
        const auto file=std::filesystem::u8path(row.at("file").get<std::string>());const uint64_t bytes=48+count*5120*4;
        if(std::filesystem::file_size(file)!=bytes)throw std::runtime_error("Vision embedding extent differs");
        std::ifstream source(file,std::ios::binary);std::array<uint32_t,4>header{};source.read(reinterpret_cast<char*>(header.data()),16);
        if(header!=std::array<uint32_t,4>{0x49564c53,1,uint32_t(count),5120})throw std::runtime_error("Unsupported native vision embedding");
        const auto digest=dsv41_boundary::sha256_stream(bytes,[&](uint64_t off,void*dst,size_t n){source.seekg(off);return bool(source.read(static_cast<char*>(dst),n));});
        if(digest!=row.at("sha256").get<std::string>())throw std::runtime_error("Projected image embedding SHA mismatch");
        result.push_back({uint32_t(position),uint32_t(count),file});previous=position+count;
    }
    return result;
}
